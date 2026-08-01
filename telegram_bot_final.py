# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 02:44:30 2026

@author: Ahmad
"""

# telegram_bot_final.py
"""
Telegram Bot for EGX Stock Analysis
===================================
Reads Excel file from GitHub (updated daily at 5 PM Egypt Time)
"""

import logging
import sys
import io
import os
import requests
import pandas as pd
from datetime import datetime
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

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# GitHub raw URL for your Excel file
GITHUB_RAW_URL = "https://raw.githubusercontent.com/ARa2of/egx-stock-analysis/main/Stock_Analysis_Output.xlsx"

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

# Cache for data
cached_data = None
cache_timestamp = None
CACHE_DURATION = 300  # 5 minutes cache

# --------------------------------------------------------------------------
# Excel Reader Functions
# --------------------------------------------------------------------------

def read_analysis_data() -> Optional[pd.DataFrame]:
    """Read the Stock_Analysis sheet from GitHub (with local fallback)."""
    global cached_data, cache_timestamp
    
    # Return cached data if fresh
    if cached_data is not None and cache_timestamp is not None:
        if (datetime.now() - cache_timestamp).total_seconds() < CACHE_DURATION:
            logger.info("📦 Using cached data")
            return cached_data
    
    try:
        # Try GitHub first
        logger.info("📥 Fetching data from GitHub...")
        response = requests.get(GITHUB_RAW_URL, timeout=10)
        
        if response.status_code == 404:
            logger.warning("⚠️ File not found on GitHub. Has the workflow run yet?")
            # Fallback to local
            return read_local_file()
        
        response.raise_for_status()
        
        df = pd.read_excel(io.BytesIO(response.content), sheet_name="Stock_Analysis")
        
        if df.empty:
            logger.warning("⚠️ Excel file is empty")
            return None
        
        logger.info(f"✅ Loaded {len(df)} stocks from GitHub")
        
        # Cache the data
        cached_data = df
        cache_timestamp = datetime.now()
        return df
        
    except Exception as e:
        logger.warning(f"⚠️ GitHub read failed: {e}")
        return read_local_file()

def read_local_file() -> Optional[pd.DataFrame]:
    """Fallback to local Excel file."""
    try:
        logger.info("📥 Falling back to local file...")
        df = pd.read_excel(LOCAL_EXCEL_FILE, sheet_name="Stock_Analysis")
        logger.info(f"✅ Loaded {len(df)} stocks from local file")
        
        cached_data = df
        cache_timestamp = datetime.now()
        return df
        
    except Exception as e2:
        logger.error(f"❌ Both GitHub and local failed: {e2}")
        return None

def get_stock_data(ticker: str) -> Optional[Dict[str, Any]]:
    """Get data for a specific ticker."""
    df = read_analysis_data()
    if df is None or df.empty:
        return None
    
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

def format_stock_response(data: Dict[str, Any]) -> str:
    """Format stock data for Telegram display."""
    rec = data.get("Recommendation", "Hold")
    emoji_map = {"Buy": "🟢", "Watch": "🟡", "Hold": "🔵", "Avoid": "🔴"}
    emoji = emoji_map.get(rec, "⚪")
    
    undervalued = data.get("Undervalued (Yes/No)", "No")
    undervalued_emoji = "✅" if undervalued == "Yes" else "❌"
    
    golden = data.get("Golden Cross (Yes/No)", "No")
    golden_emoji = "✅" if golden == "Yes" else "❌"
    
    death = data.get("Death Cross (Yes/No)", "No")
    death_emoji = "✅" if death == "Yes" else "❌"
    
    ema_bullish = data.get("EMA Bullish (50>200) (Yes/No)", "No")
    ema_emoji = "✅" if ema_bullish == "Yes" else "❌"
    
    sma50 = data.get("50 SMA")
    sma200 = data.get("200 SMA")
    sma_bullish = "✅" if (sma50 and sma200 and sma50 > sma200) else "❌"
    
    lines = [
        f"📊 *{data.get('Selected Stock', 'Unknown')}*",
        f"{'=' * 30}",
        f"📅 *Data:* {data.get('Data As Of', 'N/A')}",
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
        "",
        f"📊 *TECHNICAL INDICATORS:*",
        f"  • 50 SMA: {format_number(data.get('50 SMA'))}",
        f"  • 200 SMA: {format_number(data.get('200 SMA'))}",
        f"  • SMA50 > SMA200: {sma_bullish}",
        f"  • {golden_emoji} Golden Cross: {golden}",
        f"  • {death_emoji} Death Cross: {death}",
        f"  • 50 EMA: {format_number(data.get('50 EMA'))}",
        f"  • 200 EMA: {format_number(data.get('200 EMA'))}",
        f"  • {ema_emoji} EMA Bullish (50>200): {ema_bullish}",
        f"  • RSI: {format_number(data.get('RSI (%)'))}",
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
        "",
        f"🏆 *TAKE PROFIT TARGETS:*",
    ]
    
    tp1 = data.get('Take Profit 1')
    tp1_rr = data.get('TP1 Risk/Reward')
    tp1_pct = data.get('TP1 Reward %')
    if tp1 and not pd.isna(tp1):
        lines.append(f"  • TP1: {format_number(tp1)}  | RR: {format_number(tp1_rr)}x  | +{format_number(tp1_pct)}%")
    
    tp2 = data.get('Take Profit 2')
    tp2_rr = data.get('TP2 Risk/Reward')
    tp2_pct = data.get('TP2 Reward %')
    if tp2 and not pd.isna(tp2):
        lines.append(f"  • TP2: {format_number(tp2)}  | RR: {format_number(tp2_rr)}x  | +{format_number(tp2_pct)}%")
    
    tp3 = data.get('Take Profit 3')
    tp3_rr = data.get('TP3 Risk/Reward')
    tp3_pct = data.get('TP3 Reward %')
    if tp3 and not pd.isna(tp3):
        lines.append(f"  • TP3: {format_number(tp3)}  | RR: {format_number(tp3_rr)}x  | +{format_number(tp3_pct)}%")
    
    lines.append("")
    lines.append(f"📊 *MARKET LINKS:*")
    ticker = data.get('Selected Stock', '')
    lines.append(f"  [TradingView](https://www.tradingview.com/symbols/EGX-{ticker}/)")
    lines.append(f"  [Yahoo Finance](https://finance.yahoo.com/quote/{ticker}.CA)")
    
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

Hi {user.first_name}! I read your Excel analysis from GitHub.

📊 *Stats:* {stock_count} stocks in your portfolio
🕐 *Updated:* Daily at 5 PM Egypt Time

*Commands:*
/analyze TICKER - Get full analysis for a stock
/list - Show all stocks with quick buttons
/report - Get the latest Excel report URL
/refresh - Reload data from GitHub
/status - Check when data was last updated

*Example:*
/analyze COMI

📅 *Data As Of:* {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
    await update.message.reply_text(welcome_message, parse_mode="Markdown")

async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Get analysis for a specific ticker."""
    if not context.args:
        await update.message.reply_text(
            "❌ Please provide a ticker symbol.\n"
            "Example: /analyze COMI"
        )
        return
    
    ticker = context.args[0].upper()
    await update.message.reply_text(f"🔍 Fetching data for *{ticker}*...", parse_mode="Markdown")
    
    data = get_stock_data(ticker)
    if data is None:
        await update.message.reply_text(f"❌ Ticker '{ticker}' not found.")
        return
    
    response = format_stock_response(data)
    await update.message.reply_text(response, parse_mode="Markdown", disable_web_page_preview=True)

async def list_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show all tickers."""
    df = read_analysis_data()
    if df is None or df.empty:
        await update.message.reply_text("📋 No tickers found.")
        return
    
    lines = ["📋 *PORTFOLIO SUMMARY*", "=" * 30, ""]
    
    for rec in ["Buy", "Watch", "Hold", "Avoid"]:
        rec_df = df[df["Recommendation"] == rec]
        if not rec_df.empty:
            emoji = {"Buy": "🟢", "Watch": "🟡", "Hold": "🔵", "Avoid": "🔴"}.get(rec, "⚪")
            lines.append(f"{emoji} *{rec.upper()}* ({len(rec_df)})")
            for _, row in rec_df.iterrows():
                ticker = row["Selected Stock"]
                price = row.get("Current EGP Price")
                price_str = format_number(price) if price else "N/A"
                lines.append(f"  • {ticker} @ {price_str} EGP")
            lines.append("")
    
    lines.append("_Click a button to analyze any ticker!_")
    
    tickers = df["Selected Stock"].tolist()
    keyboard = []
    row = []
    for ticker in tickers[:30]:
        button = InlineKeyboardButton(ticker, callback_data=f"analyze_{ticker}")
        row.append(button)
        if len(row) >= 3:
            keyboard.append(row)
            row = []
    if row:
        keyboard.append(row)
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown", reply_markup=reply_markup)

async def refresh_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Refresh data from GitHub."""
    global cached_data, cache_timestamp
    
    # Clear cache
    cached_data = None
    cache_timestamp = None
    
    # Force refresh
    df = read_analysis_data()
    if df is None:
        await update.message.reply_text("❌ Could not refresh data.")
    else:
        await update.message.reply_text(f"✅ Data refreshed! {len(df)} stocks loaded.")

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
    
    if query.data.startswith("analyze_"):
        ticker = query.data.replace("analyze_", "")
        
        data = get_stock_data(ticker)
        if data is None:
            await query.edit_message_text(f"❌ Ticker '{ticker}' not found.")
            return
        
        response = format_stock_response(data)
        await query.edit_message_text(response, parse_mode="Markdown", disable_web_page_preview=True)

async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle unknown commands."""
    await update.message.reply_text(
        "❌ Unknown command.\n"
        "Use /help to see available commands."
    )

async def post_init(application: Application) -> None:
    """Show bot is ready."""
    print(f"✅ Bot started: @{application.bot.username}")
    print(f"📊 GitHub URL: {GITHUB_RAW_URL}")
    df = read_analysis_data()
    if df is not None:
        print(f"📈 Found {len(df)} stocks")
    else:
        print("⚠️ Could not load data")

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    if not BOT_TOKEN:
        print("❌ ERROR: BOT_TOKEN environment variable not set!")
        print("Set it with: export BOT_TOKEN=your_token_here")
        sys.exit(1)
    
    # Create application
    application = Application.builder().token(BOT_TOKEN).build()
    application.post_init = post_init
    
    # Add handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("analyze", analyze_command))
    application.add_handler(CommandHandler("list", list_command))
    application.add_handler(CommandHandler("refresh", refresh_command))
    application.add_handler(CommandHandler("report", report_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CallbackQueryHandler(button_callback))
    application.add_handler(MessageHandler(filters.COMMAND, unknown_command))
    
    # Start
    print("🤖 EGX Stock Analyzer Bot is starting...")
    print("📊 Reading from GitHub...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
