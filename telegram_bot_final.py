# telegram_bot_full_github.py
"""
Telegram Bot for EGX Stock Analysis - Full Version (GitHub Edition)
===================================================================

Displays ALL columns from your Stock_Analysis Excel sheet.
Reads Excel file from GitHub (updated daily at 5 PM Egypt Time)

Installation:
    pip install python-telegram-bot pandas openpyxl requests

Run:
    python telegram_bot_full_github.py
"""

import asyncio
import json
import logging
import re
import sys
import io
import os
from datetime import datetime
import pandas as pd
import requests
from typing import Optional, Dict, Any, List

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# GitHub raw URL for your Excel file
GITHUB_RAW_URL = "https://raw.githubusercontent.com/ARa2of/egx-stock-analysis/main/Stock_Analysis_Output.xlsx"
GITHUB_HISTORY_URL = "https://raw.githubusercontent.com/ARa2of/egx-stock-analysis/main/stock_history.csv"

# Get token from environment variable
BOT_TOKEN = os.environ.get("BOT_TOKEN")
if not BOT_TOKEN:
    print("❌ ERROR: BOT_TOKEN environment variable not set!")
    print("Set it with: export BOT_TOKEN=your_token_here")
    sys.exit(1)

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Gemini AI configuration
# --------------------------------------------------------------------------

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")  # cheap/fast: gemini-3.5-flash-lite

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")  # cheap/fast: gemini-3.5-flash-lite

gemini_client = None
if genai is not None and GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {e}")
elif genai is None:
    logger.warning("google-genai package not installed; /ask and /report will be disabled")
else:
    logger.warning("GEMINI_API_KEY not set; /ask and /report will be disabled")

# Columns sent to Gemini for analysis (kept compact to minimize token usage)
GEMINI_SUMMARY_COLUMNS = [
    "Selected Stock", "Index Membership", "Current EGP Price", "Recommendation",
    "Golden Cross (Yes/No)", "Death Cross (Yes/No)",
    "Diamond Cross (20>50) (Yes/No)", "RSI (%)",
    "ADX", "ADX +DI", "ADX -DI", "MFI", "BB Squeeze",
    "ChartScanAI Signal", "ChartScanAI Recommendation", "ChartScanAI Confidence",
    "Volume Multiplier (vs 1Y)", "Buy Volume Multiplier (vs 2-Month)",
    "Support", "Resistance", "Optimal Entry Price", "Stop Loss",
    "Take Profit 1", "Take Profit 2", "Take Profit 3",
    "TP1 Risk/Reward", "TP2 Risk/Reward", "TP3 Risk/Reward",
    "P/E Ratio (TTM)", "EPS (TTM)",
    "Recommendation Basis",
]

# --------------------------------------------------------------------------
# AI memory - a small local JSON store so the bot can recall prior
# snapshots/insights for a stock, and prior daily reports, without needing
# to re-query Gemini for history (saves tokens and gives real continuity).
# --------------------------------------------------------------------------

AI_MEMORY_PATH = os.environ.get(
    "AI_MEMORY_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_memory.json")
)
MAX_HISTORY_PER_TICKER = 30   # keep last N snapshots per stock
MAX_REPORT_HISTORY = 15       # keep last N daily reports

# In-memory conversation history per user (chat_id -> list of {"role", "text"}).
# Keeps the last MAX_CONVERSATION_HISTORY messages so the AI has context for
# follow-up questions.  Not persisted to disk — resets on bot restart, which
# is fine since Gemini also loses context between sessions anyway.
MAX_CONVERSATION_HISTORY = 20  # last 10 exchanges (user + bot)
_conversation_history: Dict[int, List[dict]] = {}


def _load_memory() -> dict:
    try:
        with open(AI_MEMORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"stocks": {}, "reports": []}


def _save_memory(memory: dict) -> None:
    try:
        with open(AI_MEMORY_PATH, "w", encoding="utf-8") as f:
            json.dump(memory, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"Failed to save AI memory: {e}")


def remember_stock_snapshot(ticker: str, row: dict, insight: str) -> None:
    """Save today's key figures + a short AI insight for a ticker, so a
    future question about the same stock has real history to draw on."""
    memory = _load_memory()
    today = datetime.now().strftime("%Y-%m-%d")
    entry = {
        "date": today,
        "price": row.get("Current EGP Price"),
        "recommendation": row.get("Recommendation"),
        "rsi": row.get("RSI (%)"),
        "golden_cross": row.get("Golden Cross (Yes/No)"),
        "diamond_cross": row.get("Diamond Cross (20>50) (Yes/No)"),
        "insight": (insight or "")[:500],
    }
    ticker = ticker.upper()
    history = memory.setdefault("stocks", {}).setdefault(ticker, [])
    history[:] = [h for h in history if h.get("date") != today]  # replace same-day entry
    history.append(entry)
    memory["stocks"][ticker] = history[-MAX_HISTORY_PER_TICKER:]
    _save_memory(memory)


def get_stock_history(ticker: str, limit: int = 8) -> list:
    memory = _load_memory()
    return memory.get("stocks", {}).get(ticker.upper(), [])[-limit:]


def format_stock_history(history: list) -> str:
    if not history:
        return "No prior history recorded for this stock."
    lines = [
        f"{h.get('date')}: price={h.get('price')}, rec={h.get('recommendation')}, "
        f"RSI={h.get('rsi')}, golden_cross={h.get('golden_cross')}, "
        f"diamond_cross={h.get('diamond_cross')}, prior_insight=\"{h.get('insight', '')[:150]}\""
        for h in history
    ]
    return "\n".join(lines)


def remember_report(summary: str) -> None:
    memory = _load_memory()
    memory.setdefault("reports", []).append({
        "date": datetime.now().strftime("%Y-%m-%d"),
        "summary": (summary or "")[:800],
    })
    memory["reports"] = memory["reports"][-MAX_REPORT_HISTORY:]
    _save_memory(memory)


def get_recent_reports(limit: int = 3) -> list:
    memory = _load_memory()
    return memory.get("reports", [])[-limit:]


def format_report_history(reports: list) -> str:
    if not reports:
        return "No prior daily reports recorded."
    return "\n\n".join(f"{r.get('date')}:\n{r.get('summary')}" for r in reports)


def add_to_conversation(chat_id: int, role: str, text: str) -> None:
    """Append a message to the conversation history for a chat."""
    history = _conversation_history.setdefault(chat_id, [])
    history.append({"role": role, "text": text})
    # Trim to the most recent messages
    if len(history) > MAX_CONVERSATION_HISTORY:
        _conversation_history[chat_id] = history[-MAX_CONVERSATION_HISTORY:]


def get_conversation_history(chat_id: int) -> str:
    """Format the recent conversation history for inclusion in a prompt."""
    history = _conversation_history.get(chat_id, [])
    if not history:
        return ""
    lines = ["Recent conversation (for context — continue naturally):"]
    for msg in history:
        label = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{label}: {msg['text'][:300]}")
    return "\n".join(lines)


def find_mentioned_tickers(question: str, df: pd.DataFrame) -> list:
    """Detect which tracked tickers are referenced in a free-form question,
    so we know which stocks' history to attach and which to save a new
    snapshot for."""
    tickers = df["Selected Stock"].astype(str).str.upper().unique().tolist()
    q_upper = question.upper()
    return [t for t in tickers if re.search(rf"\b{re.escape(t)}\b", q_upper)]


# --------------------------------------------------------------------------
# Wallet / Portfolio persistence
# --------------------------------------------------------------------------

WALLET_PATH = os.environ.get(
    "WALLET_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "wallet.json"),
)


def _load_wallet() -> dict:
    try:
        with open(WALLET_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_wallet(wallet: dict) -> None:
    try:
        with open(WALLET_PATH, "w", encoding="utf-8") as f:
            json.dump(wallet, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"Failed to save wallet: {e}")


def _get_chat_wallet(chat_id: int) -> dict:
    wallet = _load_wallet()
    return wallet.get(str(chat_id), {"holdings": {}})


def _save_chat_wallet(chat_id: int, data: dict) -> None:
    wallet = _load_wallet()
    wallet[str(chat_id)] = data
    _save_wallet(wallet)


def add_holding(chat_id: int, ticker: str, qty: float, avg_price: float) -> str:
    ticker = ticker.upper()
    data = _get_chat_wallet(chat_id)
    holdings = data.setdefault("holdings", {})
    if ticker in holdings:
        old = holdings[ticker]
        old_qty = old["qty"]
        old_avg = old["avg_price"]
        new_qty = old_qty + qty
        new_avg = ((old_qty * old_avg) + (qty * avg_price)) / new_qty
        holdings[ticker] = {"qty": new_qty, "avg_price": round(new_avg, 4)}
        _save_chat_wallet(chat_id, data)
        return f"Updated {ticker}: {new_qty} shares @ {new_avg:.2f} EGP (was {old_qty} @ {old_avg:.2f})"
    else:
        holdings[ticker] = {"qty": qty, "avg_price": round(avg_price, 4)}
        _save_chat_wallet(chat_id, data)
        return f"Added {ticker}: {qty} shares @ {avg_price:.2f} EGP"


def remove_holding(chat_id: int, ticker: str) -> str:
    ticker = ticker.upper()
    data = _get_chat_wallet(chat_id)
    holdings = data.get("holdings", {})
    if ticker not in holdings:
        return f"{ticker} not found in your portfolio."
    del holdings[ticker]
    _save_chat_wallet(chat_id, data)
    return f"Removed {ticker} from your portfolio."


def get_holdings(chat_id: int) -> dict:
    data = _get_chat_wallet(chat_id)
    return data.get("holdings", {})


def clear_wallet(chat_id: int) -> str:
    _save_chat_wallet(chat_id, {"holdings": {}})
    return "Portfolio cleared."


def format_portfolio(holdings: dict, df: pd.DataFrame) -> str:
    if not holdings:
        return "💰 Your portfolio is empty.\n\nUse /manage add TICKER QTY AVG_PRICE to add stocks."

    lines = ["💰 *MY PORTFOLIO*", "=" * 30, ""]
    total_invested = 0.0
    total_current = 0.0

    for ticker, info in holdings.items():
        qty = info["qty"]
        avg_price = info["avg_price"]
        invested = qty * avg_price
        total_invested += invested

        current_price = None
        if df is not None:
            row = df[df["Selected Stock"] == ticker]
            if not row.empty:
                current_price = row.iloc[0].get("Current EGP Price")
                rec = row.iloc[0].get("Recommendation", "")
                rec_emoji = {"Buy": "🟢", "Watch": "🟡", "Hold": "🔵", "Avoid": "🔴"}.get(rec, "⚪")
            else:
                rec_emoji = "⚪"
        else:
            rec_emoji = "⚪"

        if current_price is not None and not pd.isna(current_price):
            current_val = qty * current_price
            total_current += current_val
            pnl = ((current_price - avg_price) / avg_price) * 100
            pnl_emoji = "🟢" if pnl >= 0 else "🔴"
            lines.append(
                f"{rec_emoji} *{ticker}*  {qty:.0f} shares @ {avg_price:.2f}\n"
                f"   Now: {current_price:.2f} | P&L: {pnl_emoji} {pnl:+.1f}%"
            )
        else:
            lines.append(
                f"{rec_emoji} *{ticker}*  {qty:.0f} shares @ {avg_price:.2f}\n"
                f"   Current price: N/A"
            )
        lines.append("")

    if total_invested > 0 and total_current > 0:
        overall_pnl = ((total_current - total_invested) / total_invested) * 100
        pnl_emoji = "🟢" if overall_pnl >= 0 else "🔴"
        lines.append(f"💵 *Invested:* {total_invested:,.0f} EGP")
        lines.append(f"📈 *Current:* {total_current:,.0f} EGP")
        lines.append(f"{pnl_emoji} *Overall P&L:* {overall_pnl:+.1f}%")

    return "\n".join(lines)

# Cache for data
cached_data = None
cache_timestamp = None
CACHE_DURATION = 300  # 5 minutes cache

cached_history = None
cached_history_timestamp = None

cached_indices = None
cached_indices_timestamp = None

# Column name mapping for display (updated for new columns)
COLUMN_DISPLAY = {
    "Selected Stock": "📊 Stock",
    "Data As Of": "📅 Data Date",
    "Current EGP Price": "💰 EGP Price",
    "Current USD Price": "💵 USD Price",
    "Historical Min USD Price": "📉 Min USD",
    "Historical Max USD Price": "📈 Max USD",
    "Undervalued (Yes/No)": "💎 Undervalued",
    "1-Year Avg Volume": "📊 1Y Avg Vol",
    "Last Day Volume": "📊 Last Vol",
    "Volume Multiplier (vs 1Y)": "📊 Vol Multiplier",
    "Est. Buy Volume (2-Month Avg)": "📊 Buy Vol Avg",
    "Est. Buy Volume (Last Day)": "📊 Buy Vol Last",
    "Buy Volume Multiplier (vs 2-Month)": "📊 Buy Vol Multiplier",
    "Support": "📉 Support",
    "Resistance": "📈 Resistance",
    "50 SMA": "📈 50 SMA",
    "200 SMA": "📈 200 SMA",
    "Golden Cross (Yes/No)": "⭐ Golden Cross",
    "Death Cross (Yes/No)": "💀 Death Cross",
    "20 EMA": "📈 20 EMA",
    "50 EMA": "📈 50 EMA",
    "200 EMA": "📈 200 EMA",
    "EMA Bullish (50>200) (Yes/No)": "📈 EMA Bullish",
    "Diamond Cross (20>50) (Yes/No)": "💠 Diamond Cross",
    "MACD": "📉 MACD",
    "MACD Signal": "📉 MACD Signal",
    "MACD Bullish (Yes/No)": "📉 MACD Bullish",
    "P/E Ratio (TTM)": "💰 P/E Ratio (TTM)",
    "Implied Fair Value (EGP)": "🎯 Fair Value",
    "Fair Value Method": "📐 Fair Value Method",
    "Index Membership": "📇 Index Membership",
    "RSI (%)": "📊 RSI",
    "VWMA": "📊 VWMA",  # New column
    "TA Data As Of": "🕐 TA Data",  # New column
    "Optimal Entry Price": "🎯 Entry",
    "Stop Loss": "🛑 Stop Loss",
    "Stop Loss Basis": "📝 Stop Basis",
    "Take Profit 1": "🏆 TP1",
    "Take Profit 2": "🏆 TP2",
    "Take Profit 3": "🏆 TP3",
    "Take Profit Basis": "📝 TP Basis",
    "TP1 Risk/Reward": "📊 TP1 RR",
    "TP2 Risk/Reward": "📊 TP2 RR",
    "TP3 Risk/Reward": "📊 TP3 RR",
    "TP1 Reward %": "📊 TP1 %",
    "TP2 Reward %": "📊 TP2 %",
    "TP3 Reward %": "📊 TP3 %",
    "Recommendation": "🎯 Recommendation",
    "Recommendation Basis": "📝 Basis",
}

# --------------------------------------------------------------------------
# Excel Reader Functions (GitHub Version)
# --------------------------------------------------------------------------

def read_analysis_data() -> Optional[pd.DataFrame]:
    """Read the Stock_Analysis sheet from GitHub with caching."""
    global cached_data, cache_timestamp
    
    # Return cached data if fresh
    if cached_data is not None and cache_timestamp is not None:
        if (datetime.now() - cache_timestamp).total_seconds() < CACHE_DURATION:
            logger.info("📦 Using cached data")
            return cached_data
    
    try:
        # Fetch from GitHub
        logger.info("📥 Fetching data from GitHub...")
        response = requests.get(GITHUB_RAW_URL, timeout=10)
        
        if response.status_code == 404:
            logger.error(f"❌ File not found on GitHub: {GITHUB_RAW_URL}")
            return None
        
        response.raise_for_status()
        
        df = pd.read_excel(io.BytesIO(response.content), sheet_name="Stock_Analysis")
        
        if df.empty:
            logger.warning("⚠️ Excel file is empty")
            return None
        
        logger.info(f"✅ Loaded {len(df)} stocks from GitHub")
        
        # Check for required columns
        required_cols = ["Selected Stock", "Recommendation"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"⚠️ Missing columns: {missing_cols}")
        
        # Cache the data
        cached_data = df
        cache_timestamp = datetime.now()
        return df
        
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Network error: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ Error reading Excel: {e}")
        return None

def read_indices_data() -> Optional[pd.DataFrame]:
    """Read the Indices sheet (EGX30/EGX70/EGX33 snapshot) from GitHub with caching."""
    global cached_indices, cached_indices_timestamp

    if cached_indices is not None and cached_indices_timestamp is not None:
        if (datetime.now() - cached_indices_timestamp).total_seconds() < CACHE_DURATION:
            return cached_indices

    try:
        response = requests.get(GITHUB_RAW_URL, timeout=10)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        df = pd.read_excel(io.BytesIO(response.content), sheet_name="Indices")
        cached_indices = df
        cached_indices_timestamp = datetime.now()
        return df
    except Exception as e:
        logger.warning(f"Could not read Indices sheet: {e}")
        return None

def read_history_data() -> Optional[pd.DataFrame]:
    """Read the accumulated daily history CSV from GitHub with caching.
    Returns None (quietly) if the file doesn't exist yet - it's only created
    once the analysis script has run with the history-archiving feature."""
    global cached_history, cached_history_timestamp

    if cached_history is not None and cached_history_timestamp is not None:
        if (datetime.now() - cached_history_timestamp).total_seconds() < CACHE_DURATION:
            return cached_history

    try:
        response = requests.get(GITHUB_HISTORY_URL, timeout=10)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        cached_history = df
        cached_history_timestamp = datetime.now()
        return df
    except Exception as e:
        logger.warning(f"Could not read history CSV: {e}")
        return None



def format_github_history(ticker: str, limit: int = 30) -> str:
    """Format the last `limit` archived days for a ticker from the GitHub
    history CSV - the objective day-by-day record, independent of whether
    it was ever asked about before."""
    df = read_history_data()
    if df is None or df.empty or "Selected Stock" not in df.columns:
        return ""
    rows = df[df["Selected Stock"].astype(str).str.upper() == ticker.upper()]
    if rows.empty:
        return ""
    rows = rows.sort_values("Analysis Run Date").tail(limit)
    cols = [c for c in rows.columns if c != "Selected Stock"]
    return rows[cols].to_csv(index=False)

def get_stock_data(ticker: str) -> Optional[Dict[str, Any]]:
    """Get data for a specific ticker from the Excel file."""
    df = read_analysis_data()
    if df is None or df.empty:
        return None
    
    # Find the row for this ticker
    mask = df["Selected Stock"].str.upper() == ticker.upper()
    if not mask.any():
        return None
    
    row = df[mask].iloc[0]
    return row.to_dict()

def get_all_tickers() -> list:
    """Get all tickers from the Excel file."""
    df = read_analysis_data()
    if df is None or df.empty:
        return []
    return df["Selected Stock"].tolist()

def _escape_md(text: str) -> str:
    """Escape Markdown special characters for Telegram MarkdownV1."""
    if not text:
        return text
    for ch in ("_", "*", "`", "["):
        text = text.replace(ch, f"\\{ch}")
    return text

def format_number(val, decimals=2):
    """Format a number for display."""
    if val is None or pd.isna(val):
        return "N/A"
    try:
        if isinstance(val, (int, float)):
            return f"{float(val):,.{decimals}f}"
        return str(val)
    except:
        return str(val)

# --------------------------------------------------------------------------
# Gemini AI helpers
# --------------------------------------------------------------------------

def build_compact_data_summary(df: pd.DataFrame, max_basis_chars: int = 100) -> str:
    """Build a compact CSV of the key columns for all stocks, to keep the
    Gemini prompt (and token cost) small while still giving it everything
    it needs to reason about the portfolio."""
    cols = [c for c in GEMINI_SUMMARY_COLUMNS if c in df.columns]
    compact = df[cols].copy()
    if "Recommendation Basis" in compact.columns:
        compact["Recommendation Basis"] = (
            compact["Recommendation Basis"].astype(str).str.slice(0, max_basis_chars)
        )
    return compact.to_csv(index=False)


def build_index_summary() -> str:
    """Compact CSV of the EGX30/EGX70/EGX33 index snapshot, so the AI can
    weigh a stock's own index performance alongside its individual data.
    Returns "" if the Indices sheet isn't available (e.g. older Excel file)."""
    df = read_indices_data()
    if df is None or df.empty:
        return ""
    return df.to_csv(index=False)


def build_ask_prompt(data_csv: str, question: str, history_text: str = "",
                     index_csv: str = "", conversation: str = "") -> str:
    history_section = (
        f"\nPRIOR HISTORY FOR STOCKS MENTIONED IN THE QUESTION:\n{history_text}\n"
        if history_text else ""
    )
    index_section = (
        f"\nEGX INDEX SNAPSHOT (EGX30/EGX70/EGX33):\n{index_csv}\n"
        if index_csv else ""
    )
    conv_section = (
        f"\n{conversation}\n"
        if conversation else ""
    )
    return (
        "You are an experienced EGX stock analyst and trading advisor. "
        "Blend the provided data with your general market knowledge to give "
        "actionable, thoughtful advice. Don't just recite numbers — interpret "
        "them, explain what they mean, and give your honest opinion.\n\n"
        f"DATA:\n{data_csv}\n"
        f"{history_section}"
        f"{index_section}"
        f"{conv_section}\n"
        f"QUESTION: {question}\n\n"
        "Rules:\n"
        "- Use the DATA for specific numbers and facts. Don't invent figures.\n"
        "- IGNORE Fair Value, Implied Fair Value, Undervalued, or Fair Value Method columns.\n"
        "- Blend data with your general knowledge about trading, technical analysis, "
        "risk management, and market behaviour. Share your opinion.\n"
        "- If the user asks what to do (hold/buy/sell), give a clear recommendation "
        "with reasoning — don't hedge with 'it depends'.\n"
        "- If this is a follow-up question, continue the conversation naturally "
        "from the history above.\n"
        "- Be direct, honest, and practical. Max 200 words.\n"
        "- Telegram formatting only (bold, bullets). No tables."
    )


def build_daily_report_prompt(data_csv: str, report_history_text: str = "", searched: bool = False, index_csv: str = "", ticker_history: str = "") -> str:
    history_section = (
        f"\nPRIOR RECENT REPORTS (for context - note what changed since then):\n{report_history_text}\n"
        if report_history_text else ""
    )
    index_section = (
        f"\nEGX INDEX SNAPSHOT (EGX30/EGX70/EGX33):\n{index_csv}\n"
        if index_csv else ""
    )
    ticker_hist_section = (
        f"\nPRICE HISTORY (last 15 days for top stocks - note trends, momentum changes):\n{ticker_history}\n"
        if ticker_history else ""
    )
    search_line = (
        "Use search to check for recent news, sector trends, or macro/EGP "
        "currency context that could reinforce or undercut the technical "
        "picture for your top picks, and weigh that in your ranking.\n\n"
        if searched else ""
    )
    return (
        "You are an experienced EGX stock analyst. Review today's scan and produce "
        "a short Telegram report. Blend the data with your general market knowledge "
        "to give insightful, actionable commentary — not just numbers.\n\n"
        f"DATA:\n{data_csv}\n"
        f"{history_section}"
        f"{ticker_hist_section}"
        f"{index_section}\n"
        "Task: rank the top 5-8 stocks with strongest near-term upside. "
        "Use technical signals (Recommendation, Crosses, RSI, volume, support/"
        f"resistance, R/R, P/E, EPS) as primary evidence.\n\n{search_line}"
        "IMPORTANT: Compare today's values to prior days in the PRICE HISTORY. "
        "Note trends: is RSI improving or declining? Is the stock moving from "
        "Buy→Hold or Hold→Buy? Are volume and momentum building or fading?\n\n"
        "IGNORE Fair Value, Implied Fair Value, Undervalued, or Fair Value Method "
        "columns. Do NOT base recommendations on these. Focus on: "
        "historical trends, technical indicators (RSI, MACD, crosses, volume, "
        "support/resistance), and actual fundamentals (P/E, EPS).\n\n"
        "Use your general knowledge about market dynamics, sector rotation, "
        "risk management, and trading psychology to add context.\n\n"
        "Format (under 250 words):\n"
        "1. Top picks with 1-line rationale each.\n"
        "2. Any stocks with warning signs (1 line each).\n"
        "3. What changed vs prior days (trends, momentum shifts).\n\n"
        "Be direct. No introductions. Bold ticker names. Telegram formatting only."
    )


def call_gemini(prompt: str, use_search: bool = False) -> str:
    """Blocking call to the Gemini API - run this via asyncio.to_thread from
    async handlers so it doesn't block the bot's event loop.

    use_search=True enables Grounding with Google Search (billed per search
    query the model actually executes, on top of normal token cost) so it can
    pull in real current information. Off by default to keep cost minimal -
    most questions are well served by the Excel data + the model's general
    knowledge alone."""
    if gemini_client is None:
        raise RuntimeError(
            "Gemini isn't configured (missing GEMINI_API_KEY or google-genai package)."
        )
    config = None
    if use_search and genai_types is not None:
        config = genai_types.GenerateContentConfig(
            tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())]
        )
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL, contents=prompt, config=config
    )
    return response.text


async def send_long_message(send_func, text: str, chunk_size: int = 3500) -> None:
    """Send a long message in Telegram-safe chunks (4096 char hard limit).
    Falls back to plain text per chunk if Markdown parsing fails, since
    AI-generated text can produce unbalanced markdown."""
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        try:
            await send_func(chunk, parse_mode="Markdown")
        except Exception:
            await send_func(chunk)

def format_stock_response(data: Dict[str, Any]) -> str:
    """Format ALL stock data for Telegram display."""
    
    # Recommendation emoji
    rec = data.get("Recommendation", "Hold")
    emoji_map = {
        "Buy": "🟢",
        "Watch": "🟡",
        "Hold": "🔵",
        "Avoid": "🔴"
    }
    emoji = emoji_map.get(rec, "⚪")
    
    # Undervalued emoji
    undervalued = data.get("Undervalued (Yes/No)", "No")
    undervalued_emoji = "✅" if undervalued == "Yes" else "❌"
    
    # Cross emojis
    golden = data.get("Golden Cross (Yes/No)", "No")
    golden_emoji = "✅" if golden == "Yes" else "❌"
    
    death = data.get("Death Cross (Yes/No)", "No")
    death_emoji = "✅" if death == "Yes" else "❌"
    
    ema_bullish = data.get("EMA Bullish (50>200) (Yes/No)", "No")
    ema_emoji = "✅" if ema_bullish == "Yes" else "❌"

    diamond_cross = data.get("Diamond Cross (20>50) (Yes/No)", "No")
    diamond_emoji = "💠" if diamond_cross == "Yes" else "❌"

    macd_bullish = data.get("MACD Bullish (Yes/No)", "No")
    macd_emoji = "✅" if macd_bullish == "Yes" else "❌"

    index_membership = data.get("Index Membership") or "None"
    
    # SMA50 > SMA200 check
    sma50 = data.get("50 SMA")
    sma200 = data.get("200 SMA")
    sma_bullish = "✅" if (sma50 and sma200 and sma50 > sma200) else "❌"
    
    lines = [
        f"📊 *{data.get('Selected Stock', 'Unknown')}*",
        f"📇 *Index Membership:* {index_membership}",
        f"{'=' * 30}",
        f"📅 *Data:* {data.get('Data As Of', 'N/A')}",
        f"🕐 *TA Data:* {data.get('TA Data As Of', 'N/A')}",  # New line for TA timestamp
        "",
        f"{emoji} *RECOMMENDATION:* {rec}",
        f"🎯 *Score:* {format_number(data.get('Score'), 0)}/100"
        + (
            f"  (Trend {format_number(data.get('Score - Trend'), 0)} · "
            f"MACD {format_number(data.get('Score - MACD'), 0)} · "
            f"RSI {format_number(data.get('Score - RSI'), 0)} · "
            f"Vol {format_number(data.get('Score - Volume'), 0)} · "
            f"ADI {format_number(data.get('Score - ADI'), 0)} · "
            f"Supp {format_number(data.get('Score - Support'), 0)})"
            if data.get('Score - Trend') is not None else ""
        ),
        f"📝 *Basis:* {_escape_md(data.get('Recommendation Basis', 'N/A'))}",
        "",
        f"💰 *PRICES:*",
        f"  • EGP: {format_number(data.get('Current EGP Price'))}",
        f"  • USD: {format_number(data.get('Current USD Price'))}",
        f"  • Min USD: {format_number(data.get('Historical Min USD Price'))}",
        f"  • Max USD: {format_number(data.get('Historical Max USD Price'))}",
        f"  • {undervalued_emoji} Undervalued: {undervalued}",
        f"  • 🎯 Fair Value: {format_number(data.get('Implied Fair Value (EGP)'))} EGP ({data.get('Fair Value Method', 'N/A')})",
        f"  • 💰 P/E Ratio (TTM): {format_number(data.get('P/E Ratio (TTM)'))}",
        "",
        f"📊 *TECHNICAL INDICATORS:*",
        f"  • 50 SMA: {format_number(data.get('50 SMA'))}",
        f"  • 200 SMA: {format_number(data.get('200 SMA'))}",
        f"  • SMA50 > SMA200: {sma_bullish}",
        f"  • {golden_emoji} Golden Cross: {golden}",
        f"  • {death_emoji} Death Cross: {death}",
        f"  • 20 EMA: {format_number(data.get('20 EMA'))}",
        f"  • 50 EMA: {format_number(data.get('50 EMA'))}",
        f"  • 200 EMA: {format_number(data.get('200 EMA'))}",
        f"  • {ema_emoji} EMA Bullish (50>200): {ema_bullish}",
        f"  • {diamond_emoji} Diamond Cross (20>50): {diamond_cross}",
        f"  • RSI: {format_number(data.get('RSI (%)'))}",
        f"  • {'📈 Accumulation' if (data.get('ADL') or 0) > 0 else '📉 Distribution' if (data.get('ADL') or 0) < 0 else '— ADL: N/A'}",
        "",
        f"🤖 *CHARTSCAN AI:*",
        f"  • Recommendation: {'🟢 Buy' if data.get('ChartScanAI Recommendation') == 'Buy' else '🔴 Avoid' if data.get('ChartScanAI Recommendation') == 'Avoid' else '🔵 Hold' if data.get('ChartScanAI Recommendation') == 'Hold' else 'N/A'}",
        f"  • Signal: {data.get('ChartScanAI Signal', 'N/A')}",
        f"  • Confidence: {data.get('ChartScanAI Confidence', 0):.0%}" if data.get('ChartScanAI Confidence') is not None else "  • Confidence: N/A",
        f"  • Patterns: {format_number(data.get('ChartScanAI Buy Patterns'), 0)} buy / {format_number(data.get('ChartScanAI Sell Patterns'), 0)} sell",
        "",
        f"📊 *SUPPORT/RESISTANCE:*",
        f"  • Support: {format_number(data.get('Support'))}",
        f"  • Resistance: {format_number(data.get('Resistance'))}",
        "",
        f"📊 *VOLUME ANALYSIS:*",
        f"  • 1Y Avg Vol: {format_number(data.get('1-Year Avg Volume'), 0)}",
        f"  • Last Day Vol: {format_number(data.get('Last Day Volume'), 0)}",
        f"  • Vol Multiplier: {format_number(data.get('Volume Multiplier (vs 1Y)'))}x",
        f"  • Buy Vol Avg (2M): {format_number(data.get('Est. Buy Volume (2-Month Avg)'), 0)}",
        f"  • Buy Vol Last Day: {format_number(data.get('Est. Buy Volume (Last Day)'), 0)}",
        f"  • Buy Vol Multiplier: {format_number(data.get('Buy Volume Multiplier (vs 2-Month)'))}x",
        "",
        f"🎯 *TRADE SETUP:*",
        f"  • Entry: {format_number(data.get('Optimal Entry Price'))}",
        f"  • Stop Loss: {format_number(data.get('Stop Loss'))}",
        f"  • Stop Basis: {_escape_md(data.get('Stop Loss Basis', 'N/A'))}",
        "",
        f"🏆 *TAKE PROFIT TARGETS:*",
    ]
    
    # TP1
    tp1 = data.get('Take Profit 1')
    tp1_rr = data.get('TP1 Risk/Reward')
    tp1_pct = data.get('TP1 Reward %')
    if tp1 and not pd.isna(tp1):
        lines.append(f"  • TP1: {format_number(tp1)}  | RR: {format_number(tp1_rr)}x  | +{format_number(tp1_pct)}%")
    
    # TP2
    tp2 = data.get('Take Profit 2')
    tp2_rr = data.get('TP2 Risk/Reward')
    tp2_pct = data.get('TP2 Reward %')
    if tp2 and not pd.isna(tp2):
        lines.append(f"  • TP2: {format_number(tp2)}  | RR: {format_number(tp2_rr)}x  | +{format_number(tp2_pct)}%")
    
    # TP3
    tp3 = data.get('Take Profit 3')
    tp3_rr = data.get('TP3 Risk/Reward')
    tp3_pct = data.get('TP3 Reward %')
    if tp3 and not pd.isna(tp3):
        lines.append(f"  • TP3: {format_number(tp3)}  | RR: {format_number(tp3_rr)}x  | +{format_number(tp3_pct)}%")
    
    if tp1 and not pd.isna(tp1):
        lines.append(f"  📝 TP Basis: {_escape_md(data.get('Take Profit Basis', 'N/A'))}")
    
    # Add market links
    ticker = data.get('Selected Stock', '')
    lines.append("")   
    return "\n".join(lines)

# --------------------------------------------------------------------------
# Telegram Bot Handlers
# --------------------------------------------------------------------------

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send welcome message."""
    user = update.effective_user
    df = read_analysis_data()
    stock_count = len(df) if df is not None else 0
    
    welcome_message = f"""
👋 *Welcome to the EGX Stock Analyzer Bot!*

Hi {user.first_name}! I read your Excel analysis and provide stock recommendations.

📊 *Stats:* {stock_count} stocks tracked
🕐 *Updated:* Daily at 5 PM Egypt Time

*Commands:*
/show TICKER - Full analysis for a stock
/list - All stocks grouped by recommendation
/indices - EGX30/EGX70/EGX33 snapshot
/chartscan TICKER - Run YOLOv8 candlestick pattern detection
/ask <question> - Ask AI about stocks
/aireport - AI shortlist of strongest stocks
/manage - Your portfolio (add holdings, get advice)
/report - Download Excel file
/refresh - Reload data
/status - Cache status
/help - Detailed help

*Quick start:*
/manage add COMI 100 45.50
/manage advice

📅 *Last Update:* {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
    await update.message.reply_text(welcome_message, parse_mode="Markdown")

async def show_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Get full analysis for a specific ticker."""
    if not context.args:
        await update.message.reply_text(
            "❌ Please provide a ticker symbol.\n"
            "Example: /show COMI"
        )
        return
    
    ticker = context.args[0].upper()
    await update.message.reply_text(f"🔍 Fetching data for *{ticker}*...", parse_mode="Markdown")
    
    data = get_stock_data(ticker)
    if data is None:
        await update.message.reply_text(f"❌ Ticker '{ticker}' not found in the Excel file.")
        return
    
    response = format_stock_response(data)
    await update.message.reply_text(response, parse_mode="Markdown", disable_web_page_preview=True)

def classify_display_recommendation(base_rec: Optional[str], cs_rec: Optional[str]) -> tuple:
    """
    Combine the base (indicator) recommendation with the ChartScanAI
    secondary signal into a single display category for /list:
      - Strong Buy: both methods say Buy
      - Buy: exactly one method says Buy (the other method is noted as
        the conflicting/non-confirming one)
      - otherwise: falls back to the base recommendation (Watch/Avoid)

    A stock is classified into exactly ONE bucket - never both Strong Buy
    and Buy - so it's listed once, not duplicated.

    Returns (category, note) where note is a short string explaining any
    conflict/lack of secondary confirmation, or None when there's nothing
    to flag (e.g. Strong Buy, or a plain Watch/Avoid with no Buy signal
    from either method).
    """
    base_rec = base_rec or "Avoid"
    base_buy = base_rec == "Buy"
    cs_buy = cs_rec == "Buy"

    if base_buy and cs_buy:
        return "Strong Buy", None

    if base_buy and not cs_buy:
        if cs_rec == "Avoid":
            return "Buy", "⚠️ ChartScanAI says Avoid"
        elif cs_rec == "Hold":
            return "Buy", "ChartScanAI neutral"
        else:
            return "Buy", "ChartScanAI: no signal"

    if cs_buy and not base_buy:
        return "Buy", f"⚠️ base indicators say {base_rec}"

    # Neither method says Buy - use the base recommendation as-is.
    return base_rec, None


async def list_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show all tickers with quick analysis buttons."""
    df = read_analysis_data()
    if df is None or df.empty:
        await update.message.reply_text("📋 No tickers found in the Excel file.")
        return

    has_chartscan = "ChartScanAI Recommendation" in df.columns
    has_score = "Score" in df.columns

    # Classify every stock into exactly one bucket (Strong Buy / Buy / Watch
    # / Avoid), combining the base recommendation with ChartScanAI so a
    # stock never appears in more than one section.
    buckets: Dict[str, list] = {"Strong Buy": [], "Buy": [], "Watch": [], "Avoid": []}
    for _, row in df.iterrows():
        base_rec = row.get("Recommendation")
        cs_rec = row.get("ChartScanAI Recommendation") if has_chartscan else None
        category, note = classify_display_recommendation(base_rec, cs_rec)
        buckets.setdefault(category, []).append((row, note))

    lines = ["📋 *PORTFOLIO SUMMARY*", "=" * 30, ""]

    section_meta = {
        "Strong Buy": ("🔥", "STRONG BUY"),
        "Buy": ("🟢", "BUY"),
        "Watch": ("🟡", "WATCH"),
        "Avoid": ("🔴", "AVOID"),
    }

    for category in ["Strong Buy", "Buy", "Watch", "Avoid"]:
        entries = buckets.get(category, [])
        if not entries:
            continue

        # Sort by Score (base 0-100 score) when available, else leave as-is;
        # Strong Buy breaks ties by ChartScanAI confidence.
        def _sort_key(item):
            row, _ = item
            score = row.get("Score") if has_score else None
            score = score if score is not None and not pd.isna(score) else -1
            conf = row.get("ChartScanAI Confidence") if has_chartscan else None
            conf = conf if conf is not None and not pd.isna(conf) else -1
            return (score, conf)

        entries.sort(key=_sort_key, reverse=True)

        emoji, title = section_meta[category]
        lines.append(f"{emoji} *{title}* ({len(entries)})")
        for row, note in entries:
            ticker = row.get("Selected Stock", "?")
            price = row.get("Current EGP Price")
            price_str = format_number(price) if price else "N/A"
            diamond = "💠" if row.get("Diamond Cross (20>50) (Yes/No)") == "Yes" else ""
            score = row.get("Score") if has_score else None
            score_str = f" | Score: {score:.0f}/100" if score is not None and not pd.isna(score) else ""
            note_str = f" ({note})" if note else ""
            lines.append(f"  • {ticker} @ {price_str} EGP {diamond}{score_str}{note_str}")
        lines.append("")

    lines.append("_Click a button to analyze any ticker!_")
    lines.append("")
    lines.append("🔥 Strong Buy = base indicators + ChartScanAI both say Buy")
    lines.append("🟢 Buy = only ONE method says Buy (conflict shown in parentheses)")
    lines.append("🟡/🔴 Watch/Avoid = base score, no Buy signal from either method")

    # === ACCUMULATION SECTION: Buy/Watch stocks with ADL > 0 ===
    acc_entries = []
    for cat in ["Strong Buy", "Buy", "Watch"]:
        for row, note in buckets.get(cat, []):
            adl = row.get("ADL")
            if adl is not None and not pd.isna(adl) and adl > 0:
                acc_entries.append((row, note, cat))

    if acc_entries:
        def _acc_sort(item):
            row, _, _ = item
            score = row.get("Score") if has_score else None
            return score if score is not None and not pd.isna(score) else -1

        acc_entries.sort(key=_acc_sort, reverse=True)
        lines.append("")
        lines.append(f"📈 *IN ACCUMULATION* ({len(acc_entries)})")
        for row, note, orig_cat in acc_entries:
            ticker = row.get("Selected Stock", "?")
            price = row.get("Current EGP Price")
            price_str = format_number(price) if price else "N/A"
            score = row.get("Score") if has_score else None
            score_str = f" | Score: {score:.0f}/100" if score is not None and not pd.isna(score) else ""
            adl_val = row.get("ADL")
            adl_str = f" | ADL: {format_number(adl_val, 0)}" if adl_val is not None and not pd.isna(adl_val) else ""
            lines.append(f"  • {ticker} @ {price_str} EGP{score_str}{adl_str}")
        lines.append("")

    # Create buttons (limit to 30 tickers)
    tickers = df["Selected Stock"].tolist()
    keyboard = []
    row = []
    for ticker in tickers[:30]:
        button = InlineKeyboardButton(ticker, callback_data=f"show_{ticker}")
        row.append(button)
        if len(row) >= 3:
            keyboard.append(row)
            row = []
    if row:
        keyboard.append(row)

    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown", reply_markup=reply_markup)

async def report_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send the GitHub URL for the Excel report."""
    report_url = GITHUB_RAW_URL.replace("raw.githubusercontent.com", "github.com").replace("/main/", "/blob/main/")
    
    message = f"""
📊 *Excel Report Available*

📁 *Download Link:*
{report_url}

📅 *Last Updated:* Daily at 5 PM Egypt Time
🔄 *Auto-Updated:* Yes (GitHub Actions)

*Instructions:*
1. Click the link above
2. Click "Download" to save the file
3. Open with Excel to view all analysis
"""
    await update.message.reply_text(message, parse_mode="Markdown")

async def refresh_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Refresh data from Excel."""
    global cached_data, cache_timestamp
    
    # Clear cache
    cached_data = None
    cache_timestamp = None
    
    df = read_analysis_data()
    if df is None:
        await update.message.reply_text("❌ Could not read Excel file from GitHub.")
    else:
        # Count recommendations
        counts = df["Recommendation"].value_counts()
        summary = "✅ *Data Refreshed!*\n\n"
        summary += f"📊 *Total Stocks:* {len(df)}\n\n"
        summary += "*Recommendations:*\n"
        for rec in ["Buy", "Watch", "Hold", "Avoid"]:
            count = counts.get(rec, 0)
            emoji = {"Buy": "🟢", "Watch": "🟡", "Hold": "🔵", "Avoid": "🔴"}.get(rec, "⚪")
            summary += f"  {emoji} {rec}: {count}\n"
        
        # Undervalued count
        if "Undervalued (Yes/No)" in df.columns:
            undervalued = df[df["Undervalued (Yes/No)"] == "Yes"].shape[0]
            summary += f"\n💎 Undervalued: {undervalued}"
        
        if "Golden Cross (Yes/No)" in df.columns:
            summary += f"\n⭐ Golden Cross: {df[df['Golden Cross (Yes/No)'] == 'Yes'].shape[0]}"

        if "Diamond Cross (20>50) (Yes/No)" in df.columns:
            summary += f"\n💠 Diamond Cross: {df[df['Diamond Cross (20>50) (Yes/No)'] == 'Yes'].shape[0]}"
        
        await update.message.reply_text(summary, parse_mode="Markdown")

async def indices_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show the latest EGX30/EGX70/EGX33 index snapshot."""
    df = read_indices_data()
    if df is None or df.empty:
        await update.message.reply_text(
            "❌ Could not read index data. Make sure the analysis script has "
            "run at least once with index tracking enabled."
        )
        return

    lines = ["📊 *EGX Index Snapshot*\n"]
    for _, row in df.iterrows():
        if row.get("Status") != "ok":
            lines.append(f"*{row.get('Index')}*: fetch failed\n")
            continue
        macd_emoji = "✅" if row.get("MACD Bullish (Yes/No)") == "Yes" else "❌"
        lines.append(
            f"*{row.get('Index')}*\n"
            f"  • Close: {format_number(row.get('Close'))}\n"
            f"  • Change: {format_number(row.get('Change (%)'))}%\n"
            f"  • RSI: {format_number(row.get('RSI (%)'))}\n"
            f"  • {macd_emoji} MACD Bullish: {row.get('MACD Bullish (Yes/No)')}\n"
        )
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Check data status."""
    global cache_timestamp
    
    if cache_timestamp:
        last_refresh = cache_timestamp.strftime('%Y-%m-%d %H:%M:%S')
        age = (datetime.now() - cache_timestamp).total_seconds() / 60
        age_str = f"{age:.1f} minutes ago"
    else:
        last_refresh = "Never"
        age_str = "Unknown"
    
    df = read_analysis_data()
    stock_count = len(df) if df is not None else 0
    
    message = f"""
📊 *Status Report*

📁 *Data Source:* GitHub
🔄 *Last Refresh:* {last_refresh} ({age_str})
📊 *Stocks Loaded:* {stock_count}
🕐 *Scheduled Update:* Daily at 5 PM Egypt Time
💾 *Cache Duration:* 5 minutes

*Next Update:* {datetime.now().replace(hour=17, minute=0, second=0, microsecond=0).strftime('%Y-%m-%d %H:%M')} (if today) or tomorrow 5 PM
"""
    await update.message.reply_text(message, parse_mode="Markdown")

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle button clicks."""
    query = update.callback_query
    await query.answer()
    
    if query.data.startswith("show_"):
        ticker = query.data.replace("show_", "")
        
        data = get_stock_data(ticker)
        if data is None:
            await query.edit_message_text(f"❌ Ticker '{ticker}' not found.")
            return
        
        response = format_stock_response(data)
        await query.edit_message_text(response, parse_mode="Markdown", disable_web_page_preview=True)

async def ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Ask Gemini a free-form question about the current stock data.
    Any tracked ticker mentioned in the question gets its prior history
    attached for context, and a new snapshot saved after answering."""
    if gemini_client is None:
        await update.message.reply_text(
            "❌ AI analysis isn't configured yet.\n"
            "Set the GEMINI_API_KEY environment variable (and pip install google-genai) to enable /ask."
        )
        return

    if not context.args:
        await update.message.reply_text(
            "❌ Please include a question.\n"
            "Example: /ask which stocks look strongest right now?\n"
            "Example: /ask how does COMI compare to last week?\n"
            "Add 'web:' to let it check live news (small extra cost): /ask web: any recent news on COMI?"
        )
        return

    question = " ".join(context.args)

    # Search is off by default to minimize cost - opt in per-question by
    # starting with "web:" (e.g. "/ask web: any recent news on COMI?")
    use_search = False
    if question.lower().startswith("web:"):
        use_search = True
        question = question[4:].strip()

    df = read_analysis_data()
    if df is None or df.empty:
        await update.message.reply_text("❌ Could not read the latest analysis data.")
        return

    await update.message.reply_text("🤖 Thinking...")

    mentioned_tickers = find_mentioned_tickers(question, df)
    history_text = ""

    if mentioned_tickers:
        # User named specific tickers — attach their full history
        parts = []
        for t in mentioned_tickers:
            section = [f"{t}:"]
            gh_hist = format_github_history(t)
            if gh_hist:
                section.append(f"Daily archive (price/recommendation/RSI over time):\n{gh_hist}")
            ai_hist = format_stock_history(get_stock_history(t))
            section.append(f"Prior AI insights:\n{ai_hist}")
            parts.append("\n".join(section))
        history_text = "\n\n".join(parts)
    else:
        # No tickers mentioned — auto-attach history for top Buy-rated stocks
        rec_order = {"Buy": 0, "Watch": 1, "Hold": 2, "Avoid": 3}
        df_sorted = df.copy()
        df_sorted["_rec_order"] = df_sorted["Recommendation"].map(rec_order).fillna(4)
        top_tickers = df_sorted.nsmallest(5, "_rec_order")["Selected Stock"].tolist()
        parts = []
        for t in top_tickers:
            gh_hist = format_github_history(t, limit=15)
            if gh_hist:
                parts.append(f"{t}:\n{gh_hist}")
        if parts:
            history_text = "HISTORY (top-rated stocks, last 15 days):\n\n" + "\n\n".join(parts)

    data_csv = build_compact_data_summary(df)
    chat_id = update.effective_chat.id
    # Save user message and get conversation history
    add_to_conversation(chat_id, "user", question)
    conversation = get_conversation_history(chat_id)

    prompt = build_ask_prompt(data_csv, question, history_text,
                              index_csv=build_index_summary(),
                              conversation=conversation)

    try:
        answer = await asyncio.to_thread(call_gemini, prompt, use_search)
    except Exception as e:
        logger.error(f"Gemini /ask error: {e}")
        await update.message.reply_text(f"❌ AI request failed: {e}")
        return

    await send_long_message(update.message.reply_text, answer)

    # Save AI response to conversation history
    add_to_conversation(chat_id, "assistant", answer)

    # Save a fresh snapshot for any ticker discussed, so a future question
    # about it has real history (price/rec/RSI + a slice of this answer).
    for t in mentioned_tickers:
        match = df[df["Selected Stock"].astype(str).str.upper() == t]
        if not match.empty:
            remember_stock_snapshot(t, match.iloc[0].to_dict(), answer)

async def chartscan_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Run on-demand ChartScanAI analysis for a ticker."""
    if not context.args:
        await update.message.reply_text(
            "Usage: /chartscan TICKER\n"
            "Example: /chartscan COMI\n\n"
            "Runs YOLOv8 candlestick pattern detection on the latest chart."
        )
        return
    ticker = context.args[0].upper()

    await update.message.reply_text(f"🤖 Running ChartScanAI on {ticker}...")

    try:
        import yfinance as yf
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        from egx_stock_analysis import chartscan_analyze, to_yf_ticker

        # Download fresh data. EGX tickers need the ".CA" suffix on Yahoo
        # Finance (e.g. "COMI" -> "COMI.CA") - to_yf_ticker() is the same
        # helper the main analysis script uses, so this stays in sync with it.
        yf_ticker = to_yf_ticker(ticker)
        data = yf.download(yf_ticker, period="1y", progress=False, auto_adjust=True)
        if data.empty:
            await update.message.reply_text(f"❌ No data found for {ticker}")
            return

        # Flatten multi-level columns if present
        if hasattr(data.columns, 'levels') and len(data.columns.levels) > 1:
            data.columns = data.columns.get_level_values(0)

        result = chartscan_analyze(data, ticker)

        if result is None:
            await update.message.reply_text(
                f"❌ ChartScanAI analysis failed for {ticker}.\n"
                "The model may not have detected any patterns."
            )
            return

        signal = result["signal"]
        conf = result["confidence"]
        buy_p = result["buy_patterns"]
        sell_p = result["sell_patterns"]

        signal_emoji = "🟢 Buy" if signal == "Buy" else "🔴 Sell" if signal == "Sell" else "⚪ Neutral"

        # Also show indicator recommendation for comparison
        df = read_analysis_data()
        ind_rec = "N/A"
        if df is not None:
            match = df[df["Selected Stock"].str.upper() == ticker]
            if not match.empty:
                ind_rec = match.iloc[0].get("Recommendation", "N/A")

        # Agreement check
        if signal == "Buy" and ind_rec == "Buy":
            verdict = "✅ AGREE — Both methods say Buy"
        elif signal == "Sell" and ind_rec in ("Avoid", "Hold"):
            verdict = "✅ AGREE — Both methods bearish"
        elif signal == "Buy" and ind_rec in ("Avoid", "Hold"):
            verdict = "⚠️ CONFLICT — ChartScanAI says Buy, indicators disagree"
        elif signal == "Sell" and ind_rec == "Buy":
            verdict = "⚠️ CONFLICT — ChartScanAI says Sell, indicators say Buy"
        else:
            verdict = "— Mixed signals"

        text = (
            f"🤖 *ChartScanAI — {ticker}*\n"
            f"{'=' * 30}\n\n"
            f"*YOLOv8 Signal:* {signal_emoji}\n"
            f"*Confidence:* {conf:.0%}\n"
            f"*Buy patterns detected:* {buy_p}\n"
            f"*Sell patterns detected:* {sell_p}\n\n"
            f"*Indicator Recommendation:* {ind_rec}\n"
            f"*Verdict:* {verdict}"
        )
        await update.message.reply_text(text, parse_mode="Markdown")

    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)[:200]}")

async def myid_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show the current chat ID (useful for debugging/config)."""
    await update.message.reply_text(
        f"Your chat ID is: `{update.effective_chat.id}`",
        parse_mode="Markdown"
    )

async def ai_report_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """On-demand AI report: analyze the full portfolio with Gemini and reply
    with the stocks showing the strongest upside potential. Only runs when
    called - no automatic schedule, to keep token usage to a minimum."""
    global cached_data, cache_timestamp

    if gemini_client is None:
        await update.message.reply_text(
            "❌ AI analysis isn't configured yet.\n"
            "Set the GEMINI_API_KEY environment variable (and pip install google-genai) to enable /aireport."
        )
        return

    await update.message.reply_text("🤖 Analyzing today's data, one moment...")

    # Search is off by default to minimize cost - opt in with: /aireport web
    use_search = bool(context.args) and context.args[0].lower() == "web"

    # Force a fresh read so the report reflects the latest data, not a stale cache
    cached_data = None
    cache_timestamp = None
    df = read_analysis_data()
    if df is None or df.empty:
        await update.message.reply_text("❌ Could not read the latest analysis data from GitHub.")
        return

    data_csv = build_compact_data_summary(df)
    report_history_text = format_report_history(get_recent_reports())

    # Attach history for top 8 stocks by recommendation strength
    rec_order = {"Buy": 0, "Watch": 1, "Hold": 2, "Avoid": 3}
    df_sorted = df.copy()
    df_sorted["_rec_order"] = df_sorted["Recommendation"].map(rec_order).fillna(4)
    top_tickers = df_sorted.nsmallest(8, "_rec_order")["Selected Stock"].tolist()
    history_parts = []
    for t in top_tickers:
        gh_hist = format_github_history(t, limit=15)
        if gh_hist:
            history_parts.append(f"{t}:\n{gh_hist}")
    ticker_history_text = "\n\n".join(history_parts) if history_parts else ""

    prompt = build_daily_report_prompt(data_csv, report_history_text, searched=use_search, index_csv=build_index_summary(), ticker_history=ticker_history_text)

    try:
        report = await asyncio.to_thread(call_gemini, prompt, use_search)
    except Exception as e:
        logger.error(f"AI report failed: {e}")
        await update.message.reply_text(f"❌ AI report failed: {e}")
        return

    header = f"📊 *AI Stock Report* - {datetime.now().strftime('%Y-%m-%d')}\n{'=' * 30}\n\n"
    await send_long_message(update.message.reply_text, header + report)

    remember_report(report)


def build_manage_advice_prompt(data_csv: str, holdings_text: str, index_csv: str = "", ticker_history: str = "") -> str:
    index_section = (
        f"\nEGX INDEX SNAPSHOT:\n{index_csv}\n"
        if index_csv else ""
    )
    history_section = (
        f"\nHISTORY (last 15 days for your holdings - note trends, momentum):\n{ticker_history}\n"
        if ticker_history else ""
    )
    return (
        "You are an experienced portfolio advisor. The user owns specific stocks. "
        "Give brief, actionable advice (hold/buy more/sell) for EACH holding. "
        "Blend the data with your general knowledge about risk management, "
        "position sizing, and market dynamics. Be direct. No introductions.\n\n"
        f"USER'S HOLDINGS:\n{holdings_text}\n\n"
        f"MARKET DATA:\n{data_csv}\n"
        f"{history_section}"
        f"{index_section}\n"
        "IGNORE Fair Value, Implied Fair Value, Undervalued, or Fair Value Method "
        "columns. Do NOT base advice on these.\n"
        "For each holding, state: action (Hold/Buy More/Sell) + "
        "reason based on technical indicators (recommendation, RSI, crosses, "
        "volume, support/resistance, R/R) and fundamentals (P/E, EPS). "
        "Add your own assessment of risk and positioning.\n"
        "Compare today vs prior days from the HISTORY section - "
        "note if momentum is improving or declining.\n"
        "End with one overall portfolio risk note and any suggested rebalancing.\n"
        "Telegram formatting only. Under 200 words total."
    )


async def manage_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Manage portfolio holdings: /manage add|remove|show|clear|advice"""
    if not context.args:
        await update.message.reply_text(
            "💰 *Portfolio Manager*\n\n"
            "Usage:\n"
            "/manage add TICKER QTY AVG_PRICE — Add/update holding\n"
            "/manage remove TICKER — Remove a stock\n"
            "/manage show — View portfolio with P&L\n"
            "/manage clear — Remove all holdings\n"
            "/manage advice — AI advice for your portfolio",
            parse_mode="Markdown",
        )
        return

    subcommand = context.args[0].lower()
    chat_id = update.effective_chat.id

    if subcommand == "show":
        holdings = get_holdings(chat_id)
        df = read_analysis_data()
        response = format_portfolio(holdings, df)
        await update.message.reply_text(response, parse_mode="Markdown")

    elif subcommand == "clear":
        response = clear_wallet(chat_id)
        await update.message.reply_text(f"✅ {response}")

    elif subcommand == "add":
        if len(context.args) < 4:
            await update.message.reply_text(
                "Usage: /manage add TICKER QTY AVG_PRICE\n"
                "Example: /manage add COMI 100 45.50"
            )
            return
        ticker = context.args[1].upper()
        try:
            qty = float(context.args[2])
            avg_price = float(context.args[3])
        except ValueError:
            await update.message.reply_text("❌ QTY and AVG_PRICE must be numbers.")
            return
        if qty <= 0 or avg_price <= 0:
            await update.message.reply_text("❌ QTY and AVG_PRICE must be positive.")
            return
        result = add_holding(chat_id, ticker, qty, avg_price)
        await update.message.reply_text(f"✅ {result}")

    elif subcommand == "remove":
        if len(context.args) < 2:
            await update.message.reply_text("Usage: /manage remove TICKER")
            return
        ticker = context.args[1].upper()
        result = remove_holding(chat_id, ticker)
        await update.message.reply_text(f"✅ {result}")

    elif subcommand == "advice":
        holdings = get_holdings(chat_id)
        if not holdings:
            await update.message.reply_text(
                "❌ Your portfolio is empty.\n"
                "Use /manage add TICKER QTY AVG_PRICE to add stocks first."
            )
            return

        if gemini_client is None:
            await update.message.reply_text("❌ AI features disabled. Set GEMINI_API_KEY to enable.")
            return

        await update.message.reply_text("🤖 Analyzing your portfolio...")

        df = read_analysis_data()
        if df is None or df.empty:
            await update.message.reply_text("❌ No analysis data available.")
            return

        # Build holdings text with current prices
        holdings_lines = []
        for ticker, info in holdings.items():
            qty = info["qty"]
            avg_price = info["avg_price"]
            current_price = None
            if df is not None:
                row = df[df["Selected Stock"] == ticker]
                if not row.empty:
                    current_price = row.iloc[0].get("Current EGP Price")
            if current_price is not None and not pd.isna(current_price):
                pnl_pct = ((current_price - avg_price) / avg_price) * 100
                holdings_lines.append(
                    f"{ticker}: {qty:.0f} shares bought @ {avg_price:.2f}, "
                    f"now {current_price:.2f} ({pnl_pct:+.1f}%)"
                )
            else:
                holdings_lines.append(
                    f"{ticker}: {qty:.0f} shares bought @ {avg_price:.2f}, current price N/A"
                )
        holdings_text = "\n".join(holdings_lines)

        data_csv = build_compact_data_summary(df)
        index_csv = build_index_summary()

        # Attach history for each holding
        history_parts = []
        for ticker in holdings:
            gh_hist = format_github_history(ticker, limit=15)
            if gh_hist:
                history_parts.append(f"{ticker}:\n{gh_hist}")
        ticker_history = "\n\n".join(history_parts) if history_parts else ""

        prompt = build_manage_advice_prompt(data_csv, holdings_text, index_csv, ticker_history=ticker_history)

        try:
            answer = await asyncio.to_thread(call_gemini, prompt, False)
        except Exception as e:
            await update.message.reply_text(f"❌ AI advice failed: {e}")
            return

        header = f"💰 *Portfolio Advice* - {datetime.now().strftime('%Y-%m-%d')}\n{'=' * 30}\n\n"
        await send_long_message(update.message.reply_text, header + answer)

    else:
        await update.message.reply_text(
            f"❌ Unknown subcommand: {subcommand}\n"
            "Use /manage to see available options."
        )


async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle unknown commands."""
    await update.message.reply_text(
        "❌ Unknown command.\n"
        "Use /help to see available commands."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show help message."""
    help_text = (
        "📚 <b>EGX Stock Analyzer Bot - Help</b>\n\n"
        "<b>Analysis:</b>\n"
        "/show TICKER — Full analysis for one stock\n"
        "/list — All stocks grouped by recommendation\n"
        "/indices — EGX30/EGX70/EGX33 index snapshot\n"
        "/chartscan TICKER — YOLOv8 candlestick pattern detection\n\n"
        "<b>AI:</b>\n"
        "/ask &lt;question&gt; — Ask about any stock(s)\n"
        "  /ask how does COMI look?\n"
        "  /ask web: any news on SWDY? (uses live search)\n"
        "/aireport — AI shortlist of strongest stocks\n"
        "  /aireport web (with live news)\n\n"
        "<b>Portfolio:</b>\n"
        "/manage add TICKER QTY AVG_PRICE — Add holding\n"
        "/manage remove TICKER — Remove holding\n"
        "/manage show — View portfolio with P&amp;L\n"
        "/manage clear — Remove all holdings\n"
        "/manage advice — AI advice for your portfolio\n\n"
        "<b>Other:</b>\n"
        "/report — Download Excel file\n"
        "/refresh — Reload data from GitHub\n"
        "/status — Cache and update status\n\n"
        f"<b>Data:</b>\n"
        f"📁 GitHub: {GITHUB_RAW_URL}\n"
        "🔄 Auto-updated daily at 5 PM Egypt Time\n"
        "💾 Cached for 5 minutes"
    )
    await update.message.reply_text(help_text, parse_mode="HTML")

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

async def post_init(application: Application) -> None:
    """Show bot is ready."""
    print(f"✅ Bot started: @{application.bot.username}")
    print(f"📊 GitHub URL: {GITHUB_RAW_URL}")
    df = read_analysis_data()
    if df is not None:
        print(f"📈 Found {len(df)} stocks in the analysis")
        if "Recommendation" in df.columns:
            print(f"📊 Recommendations: {df['Recommendation'].value_counts().to_dict()}")
        # Check for new columns
        if "VWMA" in df.columns:
            print("✅ VWMA column found")
        if "TA Data As Of" in df.columns:
            print("✅ TA Data As Of column found")
    else:
        print("⚠️ Could not load data from GitHub")

def main():
    # Check token
    if not BOT_TOKEN:
        print("❌ ERROR: BOT_TOKEN environment variable not set!")
        print("\nTo set it:")
        print("  export BOT_TOKEN=your_token_here")
        print("\nOr create a .env file with:")
        print("  BOT_TOKEN=your_token_here")
        sys.exit(1)
    
    # Create application
    application = Application.builder().token(BOT_TOKEN).build()
    application.post_init = post_init

    if gemini_client is not None:
        print(f"✅ AI features enabled (/ask, /report) using model {GEMINI_MODEL}")
    else:
        print("⚠️ AI features disabled - set GEMINI_API_KEY (and pip install google-genai) to enable /ask and /report")
    
    # Add handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("show", show_command))
    application.add_handler(CommandHandler("list", list_command))
    application.add_handler(CommandHandler("report", report_command))
    application.add_handler(CommandHandler("refresh", refresh_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("indices", indices_command))
    application.add_handler(CommandHandler("ask", ask_command))
    application.add_handler(CommandHandler("aireport", ai_report_command))
    application.add_handler(CommandHandler("manage", manage_command))
    application.add_handler(CommandHandler("chartscan", chartscan_command))
    application.add_handler(CommandHandler("myid", myid_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CallbackQueryHandler(button_callback))
    application.add_handler(MessageHandler(filters.COMMAND, unknown_command))
    
    # Start
    print("🤖 EGX Selected Stock Analyzer Bot (Full Version - GitHub Edition) is starting...")
    print(f"📊 Reading from: {GITHUB_RAW_URL}")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
