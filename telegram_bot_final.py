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
from typing import Optional, Dict, Any

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
    "Undervalued (Yes/No)", "Implied Fair Value (EGP)", "Fair Value Method",
    "Golden Cross (Yes/No)", "Death Cross (Yes/No)",
    "Diamond Cross (20>50) (Yes/No)", "RSI (%)",
    "Volume Multiplier (vs 1Y)", "Buy Volume Multiplier (vs 2-Month)",
    "Support", "Resistance", "Optimal Entry Price", "Stop Loss",
    "Take Profit 1", "Take Profit 2", "Take Profit 3",
    "TP1 Risk/Reward", "TP2 Risk/Reward", "TP3 Risk/Reward",
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

def format_github_history(ticker: str, limit: int = 20) -> str:
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


def build_ask_prompt(data_csv: str, question: str, history_text: str = "", index_csv: str = "") -> str:
    history_section = (
        f"\nPRIOR HISTORY FOR STOCKS MENTIONED IN THE QUESTION:\n{history_text}\n"
        if history_text else ""
    )
    index_section = (
        f"\nEGX INDEX SNAPSHOT (EGX30/EGX70/EGX33):\n{index_csv}\n"
        if index_csv else ""
    )
    return (
        "You are a concise EGX stock analyst. Answer the user's question "
        "directly in 2-4 sentences max. No introductions, no fluff, no "
        "repeating the question back. Be specific with numbers.\n\n"
        f"DATA:\n{data_csv}\n"
        f"{history_section}"
        f"{index_section}\n"
        f"QUESTION: {question}\n\n"
        "Rules:\n"
        "- Use DATA as source of truth; don't invent figures.\n"
        "- Flag data-backed points vs general reasoning.\n"
        "- Keep answer under 100 words.\n"
        "- Telegram formatting only (bold, bullets). No tables."
    )


def build_daily_report_prompt(data_csv: str, report_history_text: str = "", searched: bool = False, index_csv: str = "") -> str:
    history_section = (
        f"\nPRIOR RECENT REPORTS (for context - note what changed since then):\n{report_history_text}\n"
        if report_history_text else ""
    )
    index_section = (
        f"\nEGX INDEX SNAPSHOT (EGX30/EGX70/EGX33):\n{index_csv}\n"
        if index_csv else ""
    )
    search_line = (
        "Use search to check for recent news, sector trends, or macro/EGP "
        "currency context that could reinforce or undercut the technical "
        "picture for your top picks, and weigh that in your ranking.\n\n"
        if searched else ""
    )
    return (
        "You are a concise EGX stock analyst. Review today's scan and produce "
        "a short Telegram report.\n\n"
        f"DATA:\n{data_csv}\n"
        f"{history_section}"
        f"{index_section}\n"
        "Task: rank the top 5-8 stocks with strongest near-term upside. "
        "Use technical signals (Recommendation, Crosses, RSI, volume, support/"
        f"resistance, R/R) as primary evidence.\n\n{search_line}"
        "Format (under 200 words):\n"
        "1. Top picks with 1-line rationale each.\n"
        "2. Any stocks with warning signs (1 line each).\n"
        "3. If prior reports exist: what changed (2-3 lines max).\n\n"
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
        f"📝 *Basis:* {data.get('Recommendation Basis', 'N/A')[:150]}...",
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
        f"  • MACD: {format_number(data.get('MACD'), 4)}",
        f"  • MACD Signal: {format_number(data.get('MACD Signal'), 4)}",
        f"  • {macd_emoji} MACD Bullish: {macd_bullish}",
        f"  • RSI: {format_number(data.get('RSI (%)'))}",
        f"  • VWMA: {format_number(data.get('VWMA'))}",  # New VWMA line
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
        f"  • Stop Basis: {data.get('Stop Loss Basis', 'N/A')[:80]}...",
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
        lines.append(f"  📝 TP Basis: {data.get('Take Profit Basis', 'N/A')[:100]}...")
    
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

async def list_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show all tickers with quick analysis buttons."""
    df = read_analysis_data()
    if df is None or df.empty:
        await update.message.reply_text("📋 No tickers found in the Excel file.")
        return
    
    # Create formatted list with recommendations
    lines = ["📋 *PORTFOLIO SUMMARY*", "=" * 30, ""]
    
    # Group by recommendation
    for rec in ["Buy", "Watch", "Hold", "Avoid"]:
        rec_df = df[df["Recommendation"] == rec]
        if not rec_df.empty:
            emoji = {"Buy": "🟢", "Watch": "🟡", "Hold": "🔵", "Avoid": "🔴"}.get(rec, "⚪")
            lines.append(f"{emoji} *{rec.upper()}* ({len(rec_df)})")
            for _, row in rec_df.iterrows():
                ticker = row["Selected Stock"]
                price = row.get("Current EGP Price")
                price_str = format_number(price) if price else "N/A"
                undervalued = "💎" if row.get("Undervalued (Yes/No)") == "Yes" else ""
                diamond = "💠" if row.get("Diamond Cross (20>50) (Yes/No)") == "Yes" else ""
                lines.append(f"  • {ticker} @ {price_str} EGP {undervalued}{diamond}")
            lines.append("")
    
    lines.append("_Click a button to analyze any ticker!_")
    
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

    data_csv = build_compact_data_summary(df)
    prompt = build_ask_prompt(data_csv, question, history_text, index_csv=build_index_summary())

    try:
        answer = await asyncio.to_thread(call_gemini, prompt, use_search)
    except Exception as e:
        logger.error(f"Gemini /ask error: {e}")
        await update.message.reply_text(f"❌ AI request failed: {e}")
        return

    await send_long_message(update.message.reply_text, answer)

    # Save a fresh snapshot for any ticker discussed, so a future question
    # about it has real history (price/rec/RSI + a slice of this answer).
    for t in mentioned_tickers:
        match = df[df["Selected Stock"].astype(str).str.upper() == t]
        if not match.empty:
            remember_stock_snapshot(t, match.iloc[0].to_dict(), answer)

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
    prompt = build_daily_report_prompt(data_csv, report_history_text, searched=use_search, index_csv=build_index_summary())

    try:
        report = await asyncio.to_thread(call_gemini, prompt, use_search)
    except Exception as e:
        logger.error(f"AI report failed: {e}")
        await update.message.reply_text(f"❌ AI report failed: {e}")
        return

    header = f"📊 *AI Stock Report* - {datetime.now().strftime('%Y-%m-%d')}\n{'=' * 30}\n\n"
    await send_long_message(update.message.reply_text, header + report)

    remember_report(report)


def build_manage_advice_prompt(data_csv: str, holdings_text: str, index_csv: str = "") -> str:
    index_section = (
        f"\nEGX INDEX SNAPSHOT:\n{index_csv}\n"
        if index_csv else ""
    )
    return (
        "You are a concise portfolio advisor. The user owns specific stocks. "
        "Give brief, actionable advice (hold/buy more/sell) for EACH holding. "
        "Be direct. No introductions.\n\n"
        f"USER'S HOLDINGS:\n{holdings_text}\n\n"
        f"MARKET DATA:\n{data_csv}\n"
        f"{index_section}\n"
        "For each holding, state in one line: action (Hold/Buy More/Sell) + "
        "reason based on the data (recommendation, RSI, fair value, support/"
        "resistance). If a stock looks dangerous, say so bluntly.\n"
        "End with one overall portfolio risk note (1-2 sentences max).\n"
        "Telegram formatting only. Under 150 words total."
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
        prompt = build_manage_advice_prompt(data_csv, holdings_text, index_csv)

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
    help_text = """
📚 *EGX Stock Analyzer Bot - Help*

*Analysis:*
/show TICKER — Full analysis for one stock
/list — All stocks grouped by recommendation
/indices — EGX30/EGX70/EGX33 index snapshot

*AI:*
/ask <question> — Ask about any stock(s)
  /ask how does COMI look?
  /ask web: any news on SWDY? (uses live search)
/aireport — AI shortlist of strongest stocks
  /aireport web (with live news)

*Portfolio:*
/manage add TICKER QTY AVG_PRICE — Add holding
/manage remove TICKER — Remove holding
/manage show — View portfolio with P&L
/manage clear — Remove all holdings
/manage advice — AI advice for your portfolio

*Other:*
/report — Download Excel file
/refresh — Reload data from GitHub
/status — Cache and update status

*Data:*
📁 GitHub: {url}
🔄 Auto-updated daily at 5 PM Egypt Time
💾 Cached for 5 minutes
"""
    await update.message.reply_text(help_text.format(url=GITHUB_RAW_URL), parse_mode="Markdown")

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
