# -*- coding: utf-8 -*-
"""
EGX Stock Analysis Tool (Hybrid: TradingView TA + yfinance fallback)
====================================================================

Tries to fetch indicators from TradingView (with retries & delays).
If TA fails (e.g., 429 rate limit), falls back to computing from yfinance.

All original columns are preserved: USD valuation, volume, buy volume
multiplier (vs 2â€‘month avg), support/resistance, and recommendation.

ENHANCED FEATURES:
- Enhanced entry price using multiple TradingView indicators
- Take profit levels (TP1, TP2, TP3) from resistance levels
- Risk/Reward ratios for each take profit level
- Enhanced stop loss using multiple support levels

Usage:
    python egx_stock_analysis.py input.xlsx [-o output.xlsx]
"""

import argparse
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from tradingview_ta import Interval, TradingView

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

HISTORY_PERIOD = "10y"
MIN_TRADING_DAYS = 30
EGX_SUFFIX = ".CA"
# Split/dividend-adjusted prices from yfinance. Without this, a stock split
# leaves a phantom "cliff" in the raw price history (e.g. a 2:1 split makes
# every pre-split price look 2x too high relative to today's price). That
# cliff previously fed into: the USD undervaluation check and implied fair
# value (comparing today's price to a distorted historical range), the
# Historical Min/Max USD Price columns, support/resistance swing detection,
# and the SMA/EMA/RSI/MACD fallback calculations used when TradingView data
# is unavailable for a ticker. TradingView's own indicators (used when
# available) are already split-adjusted on their end, so this specifically
# fixes our own yfinance-derived calculations.
AUTO_ADJUST = True
FRESHNESS_CHECK_PERIOD = "5d"
STALE_DATA_WARNING_DAYS = 4
FX_TICKER = "EGP=X"
EGP_SIMILARITY_BAND = 0.03
SWING_ORDER = 5
SR_LOOKBACK_DAYS = 180

SMA_SHORT_WINDOW = 50
SMA_LONG_WINDOW = 200
EMA_XSHORT_WINDOW = 20
EMA_SHORT_WINDOW = 50
EMA_LONG_WINDOW = 200
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Rolling daily history archive - lets the Telegram bot's AI features look at
# real multi-day trends for ANY stock (not just ones it's been asked about
# before), rather than only ever seeing a single day's snapshot.
HISTORY_ARCHIVE_PATH = "history.csv"
HISTORY_ARCHIVE_MAX_DAYS = 90  # keep the last N distinct trading days
HISTORY_ARCHIVE_COLUMNS = [
    "Selected Stock", "Data As Of", "Current EGP Price", "Recommendation",
    "Undervalued (Yes/No)", "Golden Cross (Yes/No)", "Death Cross (Yes/No)",
    "Diamond Cross (20>50) (Yes/No)", "RSI (%)", "Support", "Resistance",
]
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

VOLUME_SPIKE_MULTIPLIER = 1.5
NEAR_SUPPORT_PCT = 0.03
BUY_VOL_AVG_DAYS = 42          # 2-month (trading days) for buy volume average

# --- New indicator thresholds ---
ADX_TREND_THRESHOLD = 25       # ADX > 25 = strong trend
MFI_OVERBOUGHT = 80            # MFI > 80 = overbought
MFI_OVERSOLD = 20              # MFI < 20 = oversold

# Enhanced entry configuration
USE_ENHANCED_ENTRY = True       # Set to False to use original logic
ENTRY_INDICATORS = [            # Priority order for entry levels
    "Pivot.M.Fibonacci.S1",
    "Pivot.M.Classic.S1",
    "BB.lower",
    "P.SAR",
    "Ichimoku.BLine",
    "VWMA",
    "SMA50"
]

# Take profit configuration
TP_INDICATORS = [               # Priority order for take profit levels
    "Pivot.M.Classic.R1",
    "Pivot.M.Fibonacci.R1",
    "Pivot.M.Classic.R2",
    "Pivot.M.Fibonacci.R2",
    "BB.upper",
    "Pivot.M.Classic.R3",
    "Pivot.M.Fibonacci.R3",
    "SMA100",
    "SMA200"
]

# Stop loss configuration
SL_INDICATORS = [               # Priority order for stop loss levels
    "Pivot.M.Fibonacci.S1",
    "Pivot.M.Classic.S1",
    "BB.lower",
    "P.SAR",
    "SMA200"
]

# TA fetch settings - BATCHED requests (one HTTP call can carry many symbols,
# since TradingView's scanner endpoint natively supports multi-symbol queries).
# This is the real fix for 429s: it cuts ~200 individual requests down to a
# handful of chunked ones, rather than just spacing out the same volume of calls.
# Daily history archive (compact per-ticker record kept alongside the Excel
# output, so trends/history are available beyond just today's snapshot)
HISTORY_FILENAME = "stock_history.csv"
HISTORY_RETENTION_DAYS = 400  # a bit over a year, trims the file periodically

TA_EXCHANGE = "EGX"
TA_SCREENER = "egypt"
# Fundamental fields TradingView's scanner supports but the library's default
# indicator list doesn't include - requested explicitly alongside it.
TA_EXTRA_COLUMNS = ["price_earnings_ttm", "earnings_per_share_basic_ttm"]

# --------------------------------------------------------------------------
# EGX index tracking
# --------------------------------------------------------------------------

# TradingView symbols for the major EGX indices (all under the "EGX:" exchange,
# same as individual stocks, so they can be fetched with the same batch logic).
INDEX_SYMBOLS = {
    "EGX30": "EGX30",        # EGX 30 - top 30 by liquidity/activity
    "EGX70": "EGX70EWI",     # EGX 70 - equal-weighted index
    "EGX33": "SHARIAH",      # EGX 33 Shariah-compliant index
}

# Index membership (which index each stock belongs to, or UNINDEX) is read
# directly from the 'INDEX' column in the input file's Selected_Stocks sheet,
# maintained manually each quarter - see read_ticker_index_map().
TA_SYMBOLS_PER_REQUEST = 40      # symbols per single scanner request (chunked for safety)
TA_BATCH_RETRIES = 4             # retries per chunk on failure/429
TA_BATCH_RETRY_DELAY = 5         # base seconds, doubles each retry (5,10,20,40)
TA_INTER_CHUNK_DELAY_MIN = 5     # seconds to pause between chunks
TA_INTER_CHUNK_DELAY_MAX = 10

# A normal browser User-Agent instead of the library's default
# "tradingview_ta/X.X.X" string, which is an easy bot signature.
TA_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("egx_analysis")


# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------

@dataclass
class TickerData:
    raw_ticker: str
    yf_ticker: str
    history: pd.DataFrame = field(default_factory=pd.DataFrame)
    ok: bool = False
    reason: str = ""


@dataclass
class TickerTA:
    raw_ticker: str
    indicators: dict = field(default_factory=dict)
    ok: bool = False
    reason: str = ""
    fetch_time: Optional[datetime] = None  # Track when TA was fetched


# --------------------------------------------------------------------------
# Step 1: read input workbook
# --------------------------------------------------------------------------

def read_ticker_list(path: str, sheet_name: str) -> list:
    df = pd.read_excel(path, sheet_name=sheet_name, usecols=[0])
    col = df.columns[0]
    tickers = (
        df[col]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
    )
    tickers = [t for t in tickers if t]
    seen = set()
    ordered = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered


def read_ticker_index_map(path: str, sheet_name: str) -> Dict[str, str]:
    """
    Read the 'INDEX' column (column B in Selected_Stocks) mapping each ticker
    to EGX30 / EGX33 / EGX70 / UNINDEX, as maintained manually each quarter.
    Returns {} (and logs a warning) if the column doesn't exist yet - the
    script still runs fine without it, just with a blank Index Membership.
    """
    try:
        df = pd.read_excel(path, sheet_name=sheet_name)
    except Exception as e:
        log.warning("Could not read '%s' sheet for index mapping: %s", sheet_name, e)
        return {}

    if df.empty or df.shape[1] < 1:
        return {}

    ticker_col = df.columns[0]
    index_col = next((c for c in df.columns[1:] if str(c).strip().upper() == "INDEX"), None)
    if index_col is None:
        log.warning(
            "No 'INDEX' column found in '%s' - Index Membership will be blank for all stocks.",
            sheet_name
        )
        return {}

    mapping: Dict[str, str] = {}
    for _, row in df.iterrows():
        raw = row.get(ticker_col)
        if pd.isna(raw):
            continue
        ticker = str(raw).strip().upper()
        if not ticker:
            continue
        idx_val = row.get(index_col)
        mapping[ticker] = str(idx_val).strip().upper() if not pd.isna(idx_val) else "UNINDEX"
    return mapping


def to_yf_ticker(raw: str) -> str:
    raw = raw.strip().upper()
    if raw.endswith(EGX_SUFFIX):
        return raw
    return f"{raw}{EGX_SUFFIX}"


# --------------------------------------------------------------------------
# Step 2: yfinance download
# --------------------------------------------------------------------------

def download_all(tickers: list, cache: Dict[str, TickerData]) -> None:
    for raw in tickers:
        if raw in cache:
            continue
        yf_ticker = to_yf_ticker(raw)
        entry = TickerData(raw_ticker=raw, yf_ticker=yf_ticker)

        try:
            hist = yf.download(
                yf_ticker,
                period=HISTORY_PERIOD,
                interval="1d",
                auto_adjust=AUTO_ADJUST,
                progress=False,
            )
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)

            if hist.empty:
                entry.reason = "no data returned"
                log.warning("Skipping %s: %s", raw, entry.reason)
            else:
                hist = hist.dropna(subset=["Close"])
                if len(hist) < MIN_TRADING_DAYS:
                    entry.reason = f"insufficient history ({len(hist)} bars)"
                    log.warning("Skipping %s: %s", raw, entry.reason)
                else:
                    entry.history = hist
                    entry.ok = True

        except Exception as e:
            entry.reason = f"download error: {e}"
            log.warning("Skipping %s: %s", raw, entry.reason)

        cache[raw] = entry


# --------------------------------------------------------------------------
# Step 3: TradingView TA (with retry and delay)
# --------------------------------------------------------------------------

def chunk_list(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def fetch_ta_chunk_raw(symbols: List[str]) -> Dict[str, Optional[dict]]:
    """
    Manually POST to TradingView's scanner endpoint for multiple symbols in a
    single request. Done manually (rather than via tradingview_ta's own
    get_multiple_analysis helper) for two reasons:
      1. That helper doesn't check the HTTP status code before parsing the
         response as JSON, so a 429 there raises a confusing JSONDecodeError
         instead of a clean, retryable error.
      2. It lets us send a normal browser User-Agent instead of the library's
         default "tradingview_ta/X.X.X" header, an easy bot signature.

    Returns {raw_ticker: indicators_dict_or_None}.
    """
    scan_url = f"{TradingView.scan_url}{TA_SCREENER.lower()}/scan"
    tv_symbols = [f"{TA_EXCHANGE}:{s}" for s in symbols]
    # Default technical indicators (RSI, EMA/SMA, MACD, etc.) plus a couple of
    # fundamental columns TradingView's scanner supports but tradingview_ta's
    # default list doesn't include.
    indicator_columns = TradingView.indicators + TA_EXTRA_COLUMNS
    payload = {
        "symbols": {"tickers": [s.upper() for s in tv_symbols], "query": {"types": []}},
        "columns": indicator_columns,
    }
    headers = {"User-Agent": TA_USER_AGENT}
    response = requests.post(scan_url, json=payload, headers=headers, timeout=30)

    if response.status_code == 429:
        raise RuntimeError("429 rate limit")
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")

    data = response.json().get("data", [])
    by_symbol = {item["s"].upper(): item["d"] for item in data}

    result: Dict[str, Optional[dict]] = {}
    for raw, tv_symbol in zip(symbols, tv_symbols):
        row = by_symbol.get(tv_symbol.upper())
        result[raw] = dict(zip(indicator_columns, row)) if row is not None else None
    return result


def fetch_ta_batch(tickers: List[str]) -> Dict[str, "TickerTA"]:
    """
    Fetch TA data for a list of raw tickers using TradingView's multi-symbol
    scanner endpoint. Sends TA_SYMBOLS_PER_REQUEST symbols per HTTP call,
    so e.g. 200 tickers becomes ~5 requests instead of ~200 - avoiding the
    per-request rate limiting entirely rather than just slowing it down.
    """
    cache: Dict[str, TickerTA] = {}
    chunks = list(chunk_list(tickers, TA_SYMBOLS_PER_REQUEST))
    total_chunks = len(chunks)

    for idx, chunk in enumerate(chunks):
        chunk_result: Optional[Dict[str, Optional[dict]]] = None
        fetch_time = None

        for attempt in range(TA_BATCH_RETRIES):
            try:
                chunk_result = fetch_ta_chunk_raw(chunk)
                fetch_time = datetime.now()
                break
            except Exception as e:
                reason = str(e)
                is_last_attempt = attempt == TA_BATCH_RETRIES - 1
                if "429" in reason and not is_last_attempt:
                    wait = TA_BATCH_RETRY_DELAY * (2 ** attempt) + random.uniform(0, 2)
                    log.warning(
                        "TA batch %d/%d: 429 rate limit, retrying in %.1fs (attempt %d/%d)",
                        idx + 1, total_chunks, wait, attempt + 1, TA_BATCH_RETRIES
                    )
                    time.sleep(wait)
                    continue
                log.warning("TA batch %d/%d failed: %s", idx + 1, total_chunks, reason)
                chunk_result = None
                break

        for raw in chunk:
            entry = TickerTA(raw_ticker=raw)
            indicators = chunk_result.get(raw) if chunk_result else None
            if indicators and indicators.get("close") is not None:
                entry.indicators = indicators
                entry.ok = True
                entry.fetch_time = fetch_time
                log.info("%s: TA data loaded (batch %d/%d)", raw, idx + 1, total_chunks)
            else:
                entry.reason = "no TA data in batch response"
                log.warning("%s: TA fetch failed - %s", raw, entry.reason)
            cache[raw] = entry

        if idx < total_chunks - 1:
            pause = random.uniform(TA_INTER_CHUNK_DELAY_MIN, TA_INTER_CHUNK_DELAY_MAX)
            log.info(
                "Processed chunk %d/%d (%d symbols), pausing %.1fs before next chunk...",
                idx + 1, total_chunks, len(chunk), pause
            )
            time.sleep(pause)

    return cache


def fetch_all_ta(tickers: list, cache: Dict[str, TickerTA]) -> None:
    """Populate cache with TA data for all tickers via batched requests."""
    to_fetch = [t for t in tickers if t not in cache]
    if not to_fetch:
        return
    cache.update(fetch_ta_batch(to_fetch))


# --------------------------------------------------------------------------
# Step 4: FX rate (EXACT copy from egx_stock_analysis_2.py)
# --------------------------------------------------------------------------

def download_fx(period: str = HISTORY_PERIOD) -> Optional[pd.Series]:
    """Download USD/EGP exchange rate (EGP per 1 USD) as a daily Series."""
    try:
        fx = yf.download(FX_TICKER, period=period, interval="1d",
                          auto_adjust=AUTO_ADJUST, progress=False)
        if isinstance(fx.columns, pd.MultiIndex):
            fx.columns = fx.columns.get_level_values(0)
        if fx.empty:
            log.warning("No FX data returned for %s", FX_TICKER)
            return None
        return fx["Close"].dropna()
    except Exception as exc:
        log.warning("Could not download FX rate %s: %s", FX_TICKER, exc)
        return None


# --------------------------------------------------------------------------
# Fundamental data from yfinance
# --------------------------------------------------------------------------

# Reference PE for EGX stocks (emerging-market conservative average).
# Used for PE-based fair value when the stock's own trailing PE is available.
REFERENCE_PE_EGX = 12.0


def fetch_fundamentals(raw_ticker: str, yf_ticker: str, current_egp: float) -> Optional[dict]:
    """
    Fetch fundamental data from yfinance's Ticker.info endpoint.

    EGX stocks often lack ``trailingEps`` but do provide ``trailingPE``
    and ``bookValue``.  When EPS is missing we derive it from
    ``current_egp / trailingPE`` so that the Graham Number can still be
    computed.

    Returns dict with eps, book_value, pe_ratio, sector â€“ or None on failure.
    """
    try:
        ticker_obj = yf.Ticker(yf_ticker)
        info = ticker_obj.info
        if not info:
            return None

        eps = info.get("trailingEps")
        book_value = info.get("bookValue")
        pe_ratio = info.get("trailingPE")
        sector = info.get("sector", "Unknown")

        # Derive EPS from price / PE when yfinance doesn't supply it directly
        if eps is None and pe_ratio is not None and pe_ratio > 0 and current_egp is not None:
            eps = current_egp / pe_ratio
            log.debug("%s: Derived EPS %.4f from price %.2f / PE %.2f",
                      raw_ticker, eps, current_egp, pe_ratio)

        if eps is None or eps <= 0:
            log.debug("%s: No usable EPS (raw=%s, derived=%s)", raw_ticker,
                      info.get("trailingEps"), eps)
            return None

        return {
            "eps": float(eps),
            "book_value": float(book_value) if book_value is not None else None,
            "pe_ratio": float(pe_ratio) if pe_ratio is not None else None,
            "sector": sector,
        }
    except Exception as e:
        log.debug("%s: Could not fetch fundamentals: %s", raw_ticker, e)
        return None


def compute_fair_value(fundamentals: Optional[dict], current_egp: Optional[float]) -> Tuple[Optional[float], str]:
    """
    Compute fair value using Graham Number + PE-based blend.

    Graham Number = sqrt(22.5 * EPS * BookValue)
    PE-based      = EPS * Reference PE (12x for EGX emerging market)

    Returns (fair_value_egp, method_label).
    """
    if fundamentals is None:
        return None, "N/A"

    eps = fundamentals.get("eps")
    book_value = fundamentals.get("book_value")

    if eps is None or eps <= 0:
        return None, "N/A (negative earnings)"

    graham_value = None
    pe_value = None

    # Graham Number (requires positive EPS and book value)
    if book_value is not None and book_value > 0:
        graham_value = (22.5 * eps * book_value) ** 0.5

    # PE-based using reference PE for EGX (not the stock's own PE, which is circular)
    pe_value = eps * REFERENCE_PE_EGX

    # Blend available methods
    if graham_value is not None:
        fair_value = (graham_value + pe_value) / 2
        method = "Blended (Graham+PE)"
    else:
        fair_value = pe_value
        method = "PE-Based"

    if fair_value <= 0:
        return None, "N/A (negative fair value)"

    return round(fair_value, 4), method


# --------------------------------------------------------------------------
# Part 2: USD valuation analysis + fundamental fair value
# --------------------------------------------------------------------------

def usd_valuation(
    raw_ticker: str,
    cache: Dict[str, TickerData],
    fx_series: Optional[pd.Series],
    ta_cache: Optional[Dict[str, TickerTA]] = None,
) -> dict:
    """
    Computes current EGP/USD prices and fundamental fair value.

    Fair value is calculated from yfinance fundamentals (Graham Number +
    PE-based blend).  Undervaluation is flagged when the current EGP price
    is below the fair value.

    Fallback: when fundamentals are unavailable, uses the historical
    USD-comparison method (find days with similar EGP price, compare USD).
    """
    result = {
        "current_egp": None, "current_usd": None,
        "hist_min_usd": None, "hist_max_usd": None,
        "undervalued": "No",
        "implied_fair_value_egp": None,
        "fair_value_method": "N/A",
        "pe_ratio_ttm": None,
        "eps_ttm": None,
    }

    entry = cache.get(raw_ticker)
    if entry is None or not entry.ok or fx_series is None:
        return result

    egp_close = entry.history["Close"]
    aligned = pd.concat(
        [egp_close.rename("egp"), fx_series.rename("fx")], axis=1, join="inner"
    ).dropna()
    if aligned.empty:
        return result

    aligned["usd"] = aligned["egp"] / aligned["fx"]

    current_egp = float(aligned["egp"].iloc[-1])
    current_usd = float(aligned["usd"].iloc[-1])
    current_fx = float(aligned["fx"].iloc[-1])
    hist_min_usd = float(aligned["usd"].min())
    hist_max_usd = float(aligned["usd"].max())

    result.update(current_egp=current_egp, current_usd=current_usd,
                   hist_min_usd=hist_min_usd, hist_max_usd=hist_max_usd)

    # --- Primary: Fundamental fair value from yfinance ---
    fundamentals = fetch_fundamentals(raw_ticker, entry.yf_ticker, current_egp)
    fair_value, method = compute_fair_value(fundamentals, current_egp)

    if fair_value is not None:
        result["implied_fair_value_egp"] = fair_value
        result["fair_value_method"] = method
        if fundamentals:
            result["pe_ratio_ttm"] = fundamentals.get("pe_ratio")
            result["eps_ttm"] = fundamentals.get("eps")
        if current_egp < fair_value:
            result["undervalued"] = "Yes"
        log.info("%s: Fair value = %.4f (%s), current = %.4f, undervalued = %s",
                 raw_ticker, fair_value, method, current_egp, result["undervalued"])
        return result

    # --- Fallback: USD-comparison method ---
    pe_ratio = None
    eps = None
    if ta_cache:
        ta_entry = ta_cache.get(raw_ticker)
        if ta_entry and ta_entry.ok:
            ind = ta_entry.indicators
            pe_ratio = ind.get("price_earnings_ttm")
            eps = ind.get("earnings_per_share_basic_ttm")
    if eps is not None:
        result["eps_ttm"] = float(eps)
    if pe_ratio is not None:
        result["pe_ratio_ttm"] = float(pe_ratio)

    lower = current_egp * (1 - EGP_SIMILARITY_BAND)
    upper = current_egp * (1 + EGP_SIMILARITY_BAND)
    comparable = aligned[(aligned["egp"] >= lower) & (aligned["egp"] <= upper)]
    comparable = comparable.iloc[:-2] if len(comparable) > 2 else comparable

    if len(comparable) >= 10:
        historical_median_usd = float(comparable["usd"].median())
        result["implied_fair_value_egp"] = historical_median_usd * current_fx
        result["fair_value_method"] = "USD-Comparison (fallback)"
        if current_usd < historical_median_usd:
            result["undervalued"] = "Yes"
        log.info("%s: Fair value = %.4f (USD-Comparison fallback), current = %.4f",
                 raw_ticker, result["implied_fair_value_egp"], current_egp)

    return result


# --------------------------------------------------------------------------
# Volume analysis (raw volume, 1â€‘year avg) - MODIFIED to use TA volume
# --------------------------------------------------------------------------

def volume_analysis(raw_ticker: str, yf_cache: Dict[str, TickerData], ta_cache: Dict[str, TickerTA]) -> dict:
    result = {"avg_vol_1y": None, "last_day_vol": None, "vol_multiplier": None}
    
    # Try to get last day volume from TradingView first
    ta_entry = ta_cache.get(raw_ticker)
    if ta_entry and ta_entry.ok:
        ta_volume = ta_entry.indicators.get("volume")
        if ta_volume is not None:
            result["last_day_vol"] = float(ta_volume)
    
    # Get historical volume data from yfinance for averages
    entry = yf_cache.get(raw_ticker)
    if entry is None or not entry.ok or "Volume" not in entry.history.columns:
        # If we have TA volume but no yfinance, return what we have
        if result["last_day_vol"] is not None:
            return result
        return result

    vol = entry.history["Volume"].dropna()
    if vol.empty:
        return result

    last_year = vol.tail(252)
    avg_1y = float(last_year.mean()) if not last_year.empty else None

    result["avg_vol_1y"] = avg_1y
    
    # If we didn't get volume from TA, fall back to yfinance
    if result["last_day_vol"] is None:
        result["last_day_vol"] = float(vol.iloc[-1])
    
    if avg_1y and avg_1y > 0:
        result["vol_multiplier"] = round(result["last_day_vol"] / avg_1y, 3)

    return result


# --------------------------------------------------------------------------
# Money Flow Volume (buy volume) â€“ 2â€‘month average
# --------------------------------------------------------------------------

def money_flow_volume_analysis(raw_ticker: str, yf_cache: Dict[str, TickerData]) -> dict:
    result = {
        "buy_vol_avg_2mo": None,
        "buy_vol_last_day": None,
        "buy_vol_multiplier": None
    }
    entry = yf_cache.get(raw_ticker)
    required = {"High", "Low", "Close", "Volume"}
    if entry is None or not entry.ok or not required.issubset(entry.history.columns):
        return result

    hist = entry.history.dropna(subset=list(required))
    if hist.empty:
        return result

    high, low, close, volume = hist["High"], hist["Low"], hist["Close"], hist["Volume"]
    day_range = high - low

    with np.errstate(divide="ignore", invalid="ignore"):
        mfm = ((close - low) - (high - close)) / day_range
    mfm = mfm.replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(-1, 1)

    buy_fraction = (mfm + 1) / 2
    est_buy_vol = volume * buy_fraction

    buy_recent = est_buy_vol.tail(BUY_VOL_AVG_DAYS)
    if len(buy_recent) < BUY_VOL_AVG_DAYS // 2:
        return result

    avg_buy = float(buy_recent.mean())
    last_buy = float(est_buy_vol.iloc[-1])

    result["buy_vol_avg_2mo"] = avg_buy
    result["buy_vol_last_day"] = last_buy
    if avg_buy > 0:
        result["buy_vol_multiplier"] = round(last_buy / avg_buy, 3)

    return result


# --------------------------------------------------------------------------
# Support & Resistance (swing points)
# --------------------------------------------------------------------------

def find_swing_points(prices: pd.Series, order: int = SWING_ORDER):
    values = prices.values
    n = len(values)
    swing_highs, swing_lows = [], []

    for i in range(order, n - order):
        window = values[i - order: i + order + 1]
        if values[i] == window.max() and np.argmax(window) == order:
            swing_highs.append((i, values[i]))
        if values[i] == window.min() and np.argmin(window) == order:
            swing_lows.append((i, values[i]))

    return swing_highs, swing_lows


def support_resistance(raw_ticker: str, yf_cache: Dict[str, TickerData]) -> dict:
    result = {"support": None, "resistance": None}
    entry = yf_cache.get(raw_ticker)
    if entry is None or not entry.ok:
        return result

    hist = entry.history.tail(SR_LOOKBACK_DAYS)
    if len(hist) < (2 * SWING_ORDER + 1):
        return result

    highs = hist["High"] if "High" in hist.columns else hist["Close"]
    lows = hist["Low"] if "Low" in hist.columns else hist["Close"]

    swing_highs, _ = find_swing_points(highs, SWING_ORDER)
    _, swing_lows = find_swing_points(lows, SWING_ORDER)

    current_price = float(hist["Close"].iloc[-1])

    above = [v for _, v in swing_highs if v >= current_price]
    resistance = min(above) if above else (swing_highs[-1][1] if swing_highs else None)

    below = [v for _, v in swing_lows if v <= current_price]
    support = max(below) if below else (swing_lows[-1][1] if swing_lows else None)

    result["support"] = float(support) if support is not None else None
    result["resistance"] = float(resistance) if resistance is not None else None
    return result


# --------------------------------------------------------------------------
# Accumulation / Distribution & Money Flow Indicators
# --------------------------------------------------------------------------

def compute_adl(history: pd.DataFrame) -> Optional[float]:
    """Compute the Accumulation/Distribution Line.

    ADL = cumsum(((Close - Low) - (High - Close)) / (High - Low) * Volume)

    Returns the latest ADL value, or None if data is insufficient.
    """
    if history is None or len(history) < 5:
        return None
    try:
        high = history["High"].values.astype(float)
        low = history["Low"].values.astype(float)
        close = history["Close"].values.astype(float)
        vol = history["Volume"].values.astype(float)
        hl_range = high - low
        # Avoid division by zero
        hl_range[hl_range == 0] = 1e-10
        clv = ((close - low) - (high - close)) / hl_range
        adl = np.cumsum(clv * vol)
        return float(adl[-1])
    except Exception:
        return None


def compute_mfi(history: pd.DataFrame, period: int = 14) -> Optional[float]:
    """Compute Money Flow Index (volume-weighted RSI).

    MFI = 100 - (100 / (1 + positive_money_flow / negative_money_flow))

    Returns the latest MFI value (0-100), or None if data is insufficient.
    """
    if history is None or len(history) < period + 2:
        return None
    try:
        tp = ((history["High"] + history["Low"] + history["Close"]) / 3).values
        vol = history["Volume"].values.astype(float)
        raw_mf = tp * vol
        pos_flow = 0.0
        neg_flow = 0.0
        for i in range(1, len(tp)):
            if tp[i] > tp[i - 1]:
                pos_flow += raw_mf[i]
            elif tp[i] < tp[i - 1]:
                neg_flow += raw_mf[i]
        if neg_flow == 0:
            return 100.0
        mfr = pos_flow / neg_flow
        mfi = 100.0 - (100.0 / (1.0 + mfr))
        return float(mfi)
    except Exception:
        return None


def compute_bb_squeeze(bb_lower: float, bb_upper: float,
                       atr: float) -> Optional[bool]:
    """Detect Bollinger Band squeeze.

    A squeeze occurs when BB width is narrower than 0.5 * ATR,
    indicating volatility contraction before a potential breakout.
    Returns True if squeeze is active, False if not, None if data missing.
    """
    if bb_lower is None or bb_upper is None or atr is None or atr <= 0:
        return None
    bb_width = bb_upper - bb_lower
    return bb_width < (0.5 * atr)


# --------------------------------------------------------------------------
# ChartScanAI - YOLOv8 Candlestick Pattern Detection
# --------------------------------------------------------------------------

_chartscan_model = None  # Singleton: loaded once, reused across all stocks

def _get_chartscan_model():
    """Load the YOLOv8 model once and cache it for all stocks."""
    global _chartscan_model
    if _chartscan_model is not None:
        return _chartscan_model
    try:
        from ultralytics import YOLO
        model_path = os.path.join(os.path.dirname(__file__), "weights", "custom_yolov8.pt")
        if not os.path.exists(model_path):
            log.warning("ChartScanAI model not found at %s", model_path)
            return None
        _chartscan_model = YOLO(model_path)
        log.info("ChartScanAI model loaded from %s", model_path)
        return _chartscan_model
    except Exception as e:
        log.warning("Failed to load ChartScanAI model: %s", e)
        return None


def chartscan_analyze(history: pd.DataFrame, ticker: str) -> Optional[dict]:
    """Run YOLOv8 candlestick pattern detection on a stock chart.

    Generates a candlestick chart from yfinance history, runs YOLO inference,
    and returns Buy/Sell signal with confidence.

    Returns dict with keys: signal, confidence, buy_patterns, sell_patterns
    or None on failure.
    """
    if history is None or len(history) < 30:
        return None
    try:
        import mplfinance as mpf
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
        from PIL import Image
        from io import BytesIO

        model = _get_chartscan_model()
        if model is None:
            return None

        # Use latest 180 candles (or all if fewer)
        chart_data = history.iloc[-180:].copy()

        # Ensure proper datetime index for mplfinance
        if not isinstance(chart_data.index, pd.DatetimeIndex):
            chart_data.index = pd.to_datetime(chart_data.index)

        # Generate candlestick chart
        fig, ax = mpf.plot(
            chart_data, type="candle", style="yahoo",
            volume=False, returnfig=True, figsize=(18, 6.5)
        )
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)

        # Run YOLO inference
        img = Image.open(buf)
        results = model.predict(img, conf=0.3, verbose=False)

        if not results or len(results) == 0:
            return None

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return {"signal": "Neutral", "confidence": 0.0,
                    "buy_patterns": 0, "sell_patterns": 0}

        # Parse class labels: class 0 = Buy, class 1 = Sell
        buy_count = 0
        sell_count = 0
        confs = []
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            confs.append(conf)
            if cls == 0:
                buy_count += 1
            elif cls == 1:
                sell_count += 1

        avg_conf = sum(confs) / len(confs) if confs else 0.0

        if buy_count > sell_count:
            signal = "Buy"
        elif sell_count > buy_count:
            signal = "Sell"
        else:
            signal = "Neutral"

        log.info("%s: ChartScanAI — signal=%s, conf=%.2f, buy=%d, sell=%d",
                 ticker, signal, avg_conf, buy_count, sell_count)

        return {
            "signal": signal,
            "confidence": round(avg_conf, 4),
            "buy_patterns": buy_count,
            "sell_patterns": sell_count,
        }
    except Exception as e:
        log.warning("ChartScanAI failed for %s: %s", ticker, e)
        return None



# --------------------------------------------------------------------------
# Enhanced Entry Price using TradingView Indicators
# --------------------------------------------------------------------------

def get_enhanced_entry_price(
    ind: dict,
    sr_support: Optional[float],
    current_price: float,
    regime: str,
    is_near_support: bool
) -> Tuple[Optional[float], str]:
    """
    Enhanced entry price using multiple TradingView indicators.
    
    Returns:
        (entry_price, entry_basis) where entry_price is the recommended entry
        or None if no entry should be taken.
    """
    if regime == "bearish" or regime == "unknown":
        return None, "no entry in bearish/unknown regime"
    
    # Collect all potential entry levels (only those below current price)
    potential_entries = []
    
    # Loop through configured entry indicators
    for indicator_name in ENTRY_INDICATORS:
        value = ind.get(indicator_name)
        if value is not None and value < current_price:
            # Map indicator name to a friendly label
            label_map = {
                "Pivot.M.Fibonacci.S1": "Fibonacci S1",
                "Pivot.M.Classic.S1": "Classic S1",
                "BB.lower": "Bollinger Lower Band",
                "P.SAR": "Parabolic SAR",
                "Ichimoku.BLine": "Ichimoku Kijun-sen",
                "VWMA": "VWMA",
                "SMA50": "50 SMA"
            }
            label = label_map.get(indicator_name, indicator_name)
            potential_entries.append((value, label))
    
    # Add swing support if near support
    if is_near_support and sr_support is not None and sr_support < current_price:
        potential_entries.append((sr_support, "Swing Support"))
    
    # If no potential entries found, return current price
    if not potential_entries:
        return current_price, "no pullback level identified; current price as reference"
    
    # Sort by price (lowest first)
    potential_entries.sort(key=lambda x: x[0])
    
    # Get RSI for context
    rsi = ind.get("RSI")
    
    # Select the best entry level based on RSI
    if rsi is not None:
        if rsi < 30:
            # Oversold - be aggressive, use the lowest level
            entry_price, basis = potential_entries[0]
            rationale = f"oversold (RSI: {rsi:.1f})"
        elif rsi < 50:
            # Neutral - use a more conservative level (middle)
            mid_idx = len(potential_entries) // 2
            entry_price, basis = potential_entries[mid_idx]
            rationale = f"neutral (RSI: {rsi:.1f})"
        else:
            # Overbought - be conservative, use the highest level (closest to price)
            entry_price, basis = potential_entries[-1]
            rationale = f"overbought (RSI: {rsi:.1f})"
    else:
        # Default: use the first (lowest) level
        entry_price, basis = potential_entries[0]
        rationale = "no RSI data available"
    
    entry_basis = f"enhanced entry at {basis} ({rationale})"
    return entry_price, entry_basis


# --------------------------------------------------------------------------
# Enhanced Stop Loss using TradingView Indicators
# --------------------------------------------------------------------------

def get_enhanced_stop_loss(
    ind: dict,
    sr_support: Optional[float],
    entry_price: float,
    current_price: float
) -> Tuple[float, str]:
    """
    Enhanced stop loss using multiple TradingView indicators.
    
    Returns:
        (stop_loss, stop_loss_basis)
    """
    # Collect potential stop levels below entry
    stop_levels = []
    
    # Loop through configured stop loss indicators
    for indicator_name in SL_INDICATORS:
        value = ind.get(indicator_name)
        if value is not None and value < entry_price:
            label_map = {
                "Pivot.M.Fibonacci.S1": "Fibonacci S1",
                "Pivot.M.Classic.S1": "Classic S1",
                "BB.lower": "Bollinger Lower Band",
                "P.SAR": "Parabolic SAR",
                "SMA200": "200 SMA"
            }
            label = label_map.get(indicator_name, indicator_name)
            stop_levels.append((value, label))
    
    # Add swing support
    if sr_support is not None and sr_support < entry_price:
        stop_levels.append((sr_support, "Swing Support"))
    
    # Select the closest stop level below entry (most conservative)
    if stop_levels:
        # Sort by price (highest first, closest to entry)
        stop_levels.sort(key=lambda x: x[0], reverse=True)
        stop_price, basis = stop_levels[0]
        return stop_price, f"stop at {basis}"
    
    # Fallback: 5% below entry
    stop_price = entry_price * 0.95
    return stop_price, "stop at 5% below entry (default)"


# --------------------------------------------------------------------------
# Take Profit Levels using TradingView Indicators
# --------------------------------------------------------------------------

def calculate_take_profit_levels(
    ind: dict,
    current_price: float,
    entry_price: float
) -> dict:
    """
    Calculate multiple take profit levels using TradingView indicators.
    
    Returns:
        {
            'take_profit_1': float,      # Conservative target
            'take_profit_2': float,      # Moderate target  
            'take_profit_3': float,      # Aggressive target
            'take_profit_basis': str,    # Explanation
        }
    """
    result = {
        "take_profit_1": None,
        "take_profit_2": None,
        "take_profit_3": None,
        "take_profit_basis": "",
    }
    
    if entry_price is None or current_price is None:
        return result
    
    # Collect all potential resistance levels above current price
    resistance_levels = []
    
    # Loop through configured take profit indicators
    for indicator_name in TP_INDICATORS:
        value = ind.get(indicator_name)
        if value is not None and value > current_price:
            label_map = {
                "Pivot.M.Classic.R1": "Classic R1",
                "Pivot.M.Classic.R2": "Classic R2",
                "Pivot.M.Classic.R3": "Classic R3",
                "Pivot.M.Fibonacci.R1": "Fibonacci R1",
                "Pivot.M.Fibonacci.R2": "Fibonacci R2",
                "Pivot.M.Fibonacci.R3": "Fibonacci R3",
                "BB.upper": "Bollinger Upper Band",
                "SMA100": "100 SMA",
                "SMA200": "200 SMA"
            }
            label = label_map.get(indicator_name, indicator_name)
            resistance_levels.append((value, label))
    
    # Sort by price (lowest first)
    resistance_levels.sort(key=lambda x: x[0])
    
    # Remove duplicates (if prices are very close)
    unique_levels = []
    seen_prices = set()
    for price, label in resistance_levels:
        # Round to 2 decimal places for comparison
        rounded = round(price, 2)
        if rounded not in seen_prices:
            seen_prices.add(rounded)
            unique_levels.append((price, label))
    
    # Select take profit levels
    if unique_levels:
        # TP1: First resistance level (most conservative)
        result["take_profit_1"] = round(unique_levels[0][0], 4)
        
        # TP2: Second resistance level (if available)
        if len(unique_levels) >= 2:
            result["take_profit_2"] = round(unique_levels[1][0], 4)
        else:
            # Fallback: 1.5 * (TP1 - entry) + TP1
            if entry_price and result["take_profit_1"]:
                target_move = result["take_profit_1"] - entry_price
                result["take_profit_2"] = round(result["take_profit_1"] + target_move * 0.5, 4)
        
        # TP3: Third resistance level or highest available (most aggressive)
        if len(unique_levels) >= 3:
            result["take_profit_3"] = round(unique_levels[2][0], 4)
        elif len(unique_levels) == 2:
            # Use a Fibonacci extension from R2
            if result["take_profit_2"] and entry_price:
                target_move = result["take_profit_2"] - entry_price
                result["take_profit_3"] = round(result["take_profit_2"] + target_move * 0.382, 4)
        else:
            # Use the highest available with a multiplier
            if unique_levels and entry_price:
                highest = unique_levels[-1][0]
                target_move = highest - entry_price
                result["take_profit_3"] = round(highest + target_move * 0.236, 4)
        
        # Build basis explanation with labels
        basis_parts = []
        if result["take_profit_1"]:
            label = unique_levels[0][1] if unique_levels else "TP1"
            basis_parts.append(f"TP1: {label} at {round(result['take_profit_1'], 2)}")
        if result["take_profit_2"]:
            label = unique_levels[1][1] if len(unique_levels) >= 2 else "Extension"
            basis_parts.append(f"TP2: {label} at {round(result['take_profit_2'], 2)}")
        if result["take_profit_3"]:
            label = unique_levels[2][1] if len(unique_levels) >= 3 else "Extension"
            basis_parts.append(f"TP3: {label} at {round(result['take_profit_3'], 2)}")
        
        result["take_profit_basis"] = "; ".join(basis_parts)
    
    return result


# --------------------------------------------------------------------------
# Risk/Reward Ratio Calculation
# --------------------------------------------------------------------------

def calculate_risk_reward(
    entry_price: float,
    stop_loss: float,
    take_profit_1: Optional[float],
    take_profit_2: Optional[float],
    take_profit_3: Optional[float]
) -> dict:
    """
    Calculate risk/reward ratio for each take profit level.
    """
    result = {
        "tp1_rr": None,
        "tp2_rr": None,
        "tp3_rr": None,
        "tp1_reward_pct": None,
        "tp2_reward_pct": None,
        "tp3_reward_pct": None
    }
    
    if entry_price is None or stop_loss is None:
        return result
    
    risk = abs(entry_price - stop_loss)
    if risk == 0:
        return result
    
    # TP1
    if take_profit_1 is not None:
        reward = abs(take_profit_1 - entry_price)
        result["tp1_rr"] = round(reward / risk, 2)
        result["tp1_reward_pct"] = round((reward / entry_price) * 100, 2)
    
    # TP2
    if take_profit_2 is not None:
        reward = abs(take_profit_2 - entry_price)
        result["tp2_rr"] = round(reward / risk, 2)
        result["tp2_reward_pct"] = round((reward / entry_price) * 100, 2)
    
    # TP3
    if take_profit_3 is not None:
        reward = abs(take_profit_3 - entry_price)
        result["tp3_rr"] = round(reward / risk, 2)
        result["tp3_reward_pct"] = round((reward / entry_price) * 100, 2)
    
    return result


# --------------------------------------------------------------------------
# Fallback: compute SMA/EMA/RSI from yfinance (if TA fails)
# --------------------------------------------------------------------------

def compute_sma_ema_rsi_from_yf(history: pd.DataFrame) -> dict:
    close = history["Close"].dropna()
    result = {
        "sma50": None, "sma200": None, "ema20": None, "ema50": None, "ema200": None,
        "rsi": None, "macd": None, "macd_signal": None,
    }

    if len(close) >= EMA_XSHORT_WINDOW:
        result["ema20"] = float(close.ewm(span=EMA_XSHORT_WINDOW, adjust=False).mean().iloc[-1])

    if len(close) >= SMA_SHORT_WINDOW:
        result["sma50"] = float(close.rolling(SMA_SHORT_WINDOW).mean().iloc[-1])
        result["ema50"] = float(close.ewm(span=EMA_SHORT_WINDOW, adjust=False).mean().iloc[-1])

    if len(close) >= SMA_LONG_WINDOW:
        result["sma200"] = float(close.rolling(SMA_LONG_WINDOW).mean().iloc[-1])
        result["ema200"] = float(close.ewm(span=EMA_LONG_WINDOW, adjust=False).mean().iloc[-1])

    if len(close) >= RSI_PERIOD + 1:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/RSI_PERIOD, min_periods=RSI_PERIOD, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/RSI_PERIOD, min_periods=RSI_PERIOD, adjust=False).mean()
        with np.errstate(divide="ignore", invalid="ignore"):
            rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        last_rsi = rsi.iloc[-1]
        if not pd.isna(last_rsi):
            result["rsi"] = float(last_rsi)

    # MACD (standard 12/26 EMA difference, 9-period signal line)
    if len(close) >= MACD_SLOW + MACD_SIGNAL:
        ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
        ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
        result["macd"] = float(macd_line.iloc[-1])
        result["macd_signal"] = float(signal_line.iloc[-1])

    return result


# --------------------------------------------------------------------------
# Enhanced Recommendation & Entry (with original logic as fallback)
# --------------------------------------------------------------------------

def build_enhanced_recommendation_and_entry(
    val: dict, 
    mf: dict, 
    sr: dict, 
    ind: dict,  # TradingView indicators
    current_price: Optional[float],
    sma_50: Optional[float], 
    sma_200: Optional[float],
    golden_cross: Optional[str], 
    death_cross: Optional[str],
    rsi: Optional[float],
    diamond_cross: Optional[str] = None,
    adx: Optional[float] = None,
    adx_plus: Optional[float] = None,
    adx_minus: Optional[float] = None,
    mfi: Optional[float] = None,
    adl: Optional[float] = None,
    bb_squeeze: Optional[bool] = None,
    cs_signal: Optional[str] = None,
) -> dict:
    """
    Enhanced recommendation with better entry price using TradingView indicators.
    Falls back to original logic if USE_ENHANCED_ENTRY is False or if TA data is missing.
    """
    reasons = []

    # === CORE SIGNALS ===
    is_undervalued = val.get("undervalued") == "Yes"
    if is_undervalued:
        reasons.append("undervalued")

    buy_multiplier = mf.get("buy_vol_multiplier")
    is_volume_spike = buy_multiplier is not None and buy_multiplier >= VOLUME_SPIKE_MULTIPLIER
    if is_volume_spike:
        reasons.append("buy-volume spike")

    support = sr.get("support")
    is_near_support = False
    if current_price is not None and support is not None and support > 0:
        distance_pct = (current_price - support) / support
        is_near_support = 0 <= distance_pct <= NEAR_SUPPORT_PCT
    if is_near_support:
        reasons.append("near support")

    signal_count = sum([is_undervalued, is_volume_spike, is_near_support])

    # === NEW INDICATOR SIGNALS ===
    # ADX: trend strength
    has_strong_trend = adx is not None and adx >= ADX_TREND_THRESHOLD
    if has_strong_trend:
        reasons.append(f"ADX {adx:.1f} (strong trend)")

    # ADX+DI vs ADX-DI: direction confirmation
    adx_bullish_dir = adx_plus is not None and adx_minus is not None and adx_plus > adx_minus
    adx_bearish_dir = adx_plus is not None and adx_minus is not None and adx_minus > adx_plus
    if adx_bullish_dir:
        reasons.append("ADX+DI > ADX-DI (bullish direction)")
    elif adx_bearish_dir:
        reasons.append("ADX-DI > ADX+DI (bearish direction)")

    # MFI: money flow
    is_mfi_overbought = mfi is not None and mfi >= MFI_OVERBOUGHT
    is_mfi_oversold = mfi is not None and mfi <= MFI_OVERSOLD
    if is_mfi_overbought:
        reasons.append(f"MFI {mfi:.1f} (overbought)")
    if is_mfi_oversold:
        reasons.append(f"MFI {mfi:.1f} (oversold)")

    # ADL: accumulation/distribution (cumulative line; positive = net accumulation)
    is_accumulating = False
    is_distributing = False
    if adl is not None:
        is_accumulating = adl > 0
        is_distributing = adl < 0
    if is_accumulating:
        reasons.append("ADL positive (accumulation)")
    elif is_distributing:
        reasons.append("ADL negative (distribution)")

    # BB Squeeze: volatility compression
    if bb_squeeze is True:
        reasons.append("BB Squeeze ON (volatility compression)")

    # === DETERMINE TREND REGIME ===
    # Use ADX direction to reinforce/override Golden/Death Cross
    if golden_cross == "Yes":
        regime = "bullish"
    elif death_cross == "Yes":
        regime = "bearish"
    elif adx_bullish_dir and has_strong_trend:
        regime = "bullish"  # ADX confirms bullish even without golden cross
    elif adx_bearish_dir and has_strong_trend:
        regime = "bearish"  # ADX confirms bearish even without death cross
    else:
        regime = "unknown"

    # === SCORING SYSTEM ===
    # Positive signals add, negative signals subtract
    bull_score = 0
    bear_score = 0

    # Base signals
    bull_score += signal_count  # undervalued, volume spike, near support

    # Trend confirmation
    if regime == "bullish":
        bull_score += 2
        reasons.append("Golden Cross confirmed" if golden_cross == "Yes" else "ADX confirms uptrend")
    elif regime == "bearish":
        bear_score += 3
        reasons.append("Death Cross (confirmed downtrend)" if death_cross == "Yes" else "ADX confirms downtrend")

    # Diamond Cross
    has_diamond = diamond_cross == "Yes"
    if has_diamond:
        bull_score += 1
        reasons.append("Diamond Cross (EMA20>EMA50)")

    # ADX strength bonus
    if has_strong_trend:
        if adx_bullish_dir:
            bull_score += 1
        elif adx_bearish_dir:
            bear_score += 1

    # Money flow
    if is_mfi_oversold and regime != "bearish":
        bull_score += 1  # oversold in non-bearish = potential bounce
    if is_mfi_overbought:
        bear_score += 1

    # Accumulation/Distribution
    if is_accumulating and regime == "bullish":
        bull_score += 1
    if is_distributing and regime != "bullish":
        bear_score += 1

    # ChartScanAI cross-validation scoring
    if cs_agrees_buy and regime == "bullish":
        bull_score += 2  # strong confirmation
    elif cs_agrees_buy:
        bull_score += 1  # possible early signal
    if cs_agrees_sell and regime == "bearish":
        bear_score += 2  # strong confirmation
    elif cs_agrees_sell:
        bear_score += 1
    if cs_conflict_buy:
        bear_score += 1  # conflict reduces confidence
    if cs_conflict_sell:
        bull_score -= 1  # conflict reduces confidence

    # BB Squeeze neutral (noted but doesn't shift score)

    # ChartScanAI cross-validation
    cs_agrees_buy = cs_signal == "Buy"
    cs_agrees_sell = cs_signal == "Sell"
    cs_conflict_buy = cs_agrees_sell and regime == "bullish"
    cs_conflict_sell = cs_agrees_buy and regime == "bearish"
    if cs_agrees_buy:
        reasons.append("ChartScanAI confirms Buy")
    if cs_agrees_sell:
        reasons.append("ChartScanAI confirms Sell")
    if cs_conflict_buy:
        reasons.append("ChartScanAI conflicts (says Sell in uptrend)")
    if cs_conflict_sell:
        reasons.append("ChartScanAI conflicts (says Buy in downtrend)")

    # === RECOMMENDATION DECISION ===
    overbought = rsi is not None and rsi >= RSI_OVERBOUGHT
    if overbought:
        reasons.append(f"RSI overbought ({rsi:.1f})")
    oversold = rsi is not None and rsi <= RSI_OVERSOLD
    if oversold:
        reasons.append(f"RSI oversold ({rsi:.1f})")

    if regime == "bearish" and bear_score >= 3:
        recommendation = "Avoid"
    elif regime == "bearish" and bear_score < 3:
        # Bearish regime but weak signals → Watch (possible reversal)
        recommendation = "Watch" if is_mfi_oversold or oversold else "Hold"
        if recommendation == "Watch":
            reasons.append("weak bearish + oversold → possible reversal")
    elif overbought:
        recommendation = "Watch"
    elif regime == "bullish":
        if signal_count >= 2 and bull_score >= 4:
            recommendation = "Buy"
        elif signal_count >= 1 and bull_score >= 3:
            recommendation = "Buy" if has_diamond else "Watch"
        elif bull_score >= 3:
            recommendation = "Watch"
        else:
            recommendation = "Hold"
    else:  # unknown
        if bull_score >= 4:
            recommendation = "Buy"
        elif bull_score >= 3:
            recommendation = "Watch"
        else:
            recommendation = "Hold"

    # Upgrade: strong ADX + accumulation in uptrend → Buy
    if (recommendation == "Watch" and has_strong_trend and adx_bullish_dir
            and is_accumulating and regime == "bullish"):
        recommendation = "Buy"
        reasons.append("ADX + accumulation upgrade to Buy")

    # === ENTRY PRICE ===
    if USE_ENHANCED_ENTRY and ind and len(ind) > 0:
        entry_price, entry_basis = get_enhanced_entry_price(
            ind, sr.get("support"), current_price, regime, is_near_support
        )
    else:
        entry_price, entry_basis = None, ""
        if regime == "bullish":
            if is_near_support and support is not None:
                entry_price = support
                entry_basis = "pullback entry at support in a confirmed uptrend"
            elif sma_50 is not None and current_price is not None and sma_50 < current_price:
                entry_price = sma_50
                entry_basis = "pullback entry at the rising 50 SMA"
            else:
                entry_price = current_price
                entry_basis = "no pullback level identified; current price as reference"
        elif regime == "bearish":
            entry_price = None
            entry_basis = "confirmed downtrend; no technical entry recommended"
        else:
            entry_price = None
            entry_basis = "trend undetermined"

    # === STOP LOSS ===
    if USE_ENHANCED_ENTRY and entry_price is not None and ind and len(ind) > 0:
        stop_loss, stop_loss_basis = get_enhanced_stop_loss(
            ind, sr.get("support"), entry_price, current_price
        )
    else:
        if entry_price is not None:
            stop_loss = entry_price * 0.95
            stop_loss_basis = "stop at 5% below entry (default)"
        else:
            stop_loss = None
            stop_loss_basis = "no stop loss (no entry)"

    # === TAKE PROFIT LEVELS ===
    if USE_ENHANCED_ENTRY and entry_price is not None and ind and len(ind) > 0:
        tp_results = calculate_take_profit_levels(ind, current_price, entry_price)
        tp1 = tp_results["take_profit_1"]
        tp2 = tp_results["take_profit_2"]
        tp3 = tp_results["take_profit_3"]
        tp_basis = tp_results["take_profit_basis"]
    else:
        tp1 = tp2 = tp3 = None
        tp_basis = "no take profit levels (insufficient TA data)"

    # === RISK/REWARD RATIOS ===
    if entry_price is not None and stop_loss is not None:
        rr_results = calculate_risk_reward(entry_price, stop_loss, tp1, tp2, tp3)
        tp1_rr = rr_results["tp1_rr"]
        tp2_rr = rr_results["tp2_rr"]
        tp3_rr = rr_results["tp3_rr"]
        tp1_reward_pct = rr_results["tp1_reward_pct"]
        tp2_reward_pct = rr_results["tp2_reward_pct"]
        tp3_reward_pct = rr_results["tp3_reward_pct"]
    else:
        tp1_rr = tp2_rr = tp3_rr = None
        tp1_reward_pct = tp2_reward_pct = tp3_reward_pct = None

    basis = ", ".join(reasons) if reasons else "no signals triggered"
    basis = f"{basis}; entry: {entry_basis}" if entry_basis else basis

    return {
        "recommendation": recommendation,
        "recommendation_basis": basis,
        "optimal_entry_price": round(entry_price, 4) if entry_price is not None else None,
        "stop_loss": round(stop_loss, 4) if stop_loss is not None else None,
        "stop_loss_basis": stop_loss_basis,
        "take_profit_1": round(tp1, 4) if tp1 is not None else None,
        "take_profit_2": round(tp2, 4) if tp2 is not None else None,
        "take_profit_3": round(tp3, 4) if tp3 is not None else None,
        "take_profit_basis": tp_basis,
        "tp1_rr": tp1_rr,
        "tp2_rr": tp2_rr,
        "tp3_rr": tp3_rr,
        "tp1_reward_pct": tp1_reward_pct,
        "tp2_reward_pct": tp2_reward_pct,
        "tp3_reward_pct": tp3_reward_pct,
    }


# --------------------------------------------------------------------------
# Daily history archive
# --------------------------------------------------------------------------

# Compact set of columns worth keeping long-term (kept small so the archive
# stays lightweight even after a year+ of daily runs).
HISTORY_COLUMNS = [
    "Analysis Run Date", "Selected Stock", "Index Membership", "Current EGP Price", "Recommendation",
    "Undervalued (Yes/No)", "Implied Fair Value (EGP)", "Fair Value Method",
    "Golden Cross (Yes/No)", "Death Cross (Yes/No)",
    "Diamond Cross (20>50) (Yes/No)", "RSI (%)",
    "ADX", "ADX +DI", "ADX -DI", "MFI", "BB Squeeze", "ADL",
    "ChartScanAI Signal", "ChartScanAI Confidence",
    "Volume Multiplier (vs 1Y)", "Buy Volume Multiplier (vs 2-Month)",
    "Support", "Resistance",
    "P/E Ratio (TTM)", "EPS (TTM)",
    "MACD Bullish (Yes/No)",
    "Optimal Entry Price", "Stop Loss",
    "Take Profit 1", "Take Profit 2", "Take Profit 3",
]


def append_daily_history(out_df: pd.DataFrame, output_path: str) -> None:
    """
    Append today's compact snapshot to a running history CSV (kept next to
    the Excel output, e.g. stock_history.csv). This is what lets the bot (or
    anyone) look back at how a stock's recommendation/RSI/price evolved over
    time, rather than only ever seeing today's snapshot.
    """
    history_path = os.path.join(os.path.dirname(os.path.abspath(output_path)) or ".", HISTORY_FILENAME)
    cols = [c for c in HISTORY_COLUMNS if c in out_df.columns]
    today_snapshot = out_df[cols].copy()

    if os.path.exists(history_path):
        try:
            existing = pd.read_csv(history_path)
        except Exception as e:
            log.warning("Could not read existing history file (%s), starting fresh: %s", history_path, e)
            existing = pd.DataFrame(columns=cols)
        combined = pd.concat([existing, today_snapshot], ignore_index=True)
        # De-dupe same ticker/date pairs (keep the latest run if the script
        # was run more than once on the same day), then trim to a max age.
        combined.drop_duplicates(subset=["Analysis Run Date", "Selected Stock"], keep="last", inplace=True)
    else:
        combined = today_snapshot

    if "Analysis Run Date" in combined.columns:
        cutoff = (datetime.now() - timedelta(days=HISTORY_RETENTION_DAYS)).strftime("%Y-%m-%d")
        combined = combined[combined["Analysis Run Date"] >= cutoff]

    combined.to_csv(history_path, index=False)
    log.info("Daily history updated: %s (%d rows)", history_path, len(combined))


def fetch_index_snapshot() -> pd.DataFrame:
    """
    Fetch current values for EGX30, EGX70, and EGX33 using the same batched
    TradingView fetch already used for individual stocks (indices are just
    symbols under the same "EGX:" exchange, so no special-casing needed).
    """
    raw_symbols = list(INDEX_SYMBOLS.values())
    ta_cache: Dict[str, TickerTA] = {}
    fetch_all_ta(raw_symbols, ta_cache)

    rows = []
    for friendly_name, symbol in INDEX_SYMBOLS.items():
        entry = ta_cache.get(symbol)
        if entry is None or not entry.ok:
            rows.append({"Index": friendly_name, "TradingView Symbol": symbol, "Status": "fetch failed"})
            continue
        ind = entry.indicators
        macd = ind.get("MACD.macd")
        macd_signal = ind.get("MACD.signal")
        rows.append({
            "Index": friendly_name,
            "TradingView Symbol": symbol,
            "Status": "ok",
            "Close": round(ind.get("close"), 2) if ind.get("close") is not None else None,
            "Change (%)": round(ind.get("change"), 2) if ind.get("change") is not None else None,
            "RSI (%)": round(ind.get("RSI"), 2) if ind.get("RSI") is not None else None,
            "50 SMA": round(ind.get("SMA50"), 2) if ind.get("SMA50") is not None else None,
            "200 SMA": round(ind.get("SMA200"), 2) if ind.get("SMA200") is not None else None,
            "20 EMA": round(ind.get("EMA20"), 2) if ind.get("EMA20") is not None else None,
            "50 EMA": round(ind.get("EMA50"), 2) if ind.get("EMA50") is not None else None,
            "200 EMA": round(ind.get("EMA200"), 2) if ind.get("EMA200") is not None else None,
            "MACD": round(macd, 2) if macd is not None else None,
            "MACD Signal": round(macd_signal, 2) if macd_signal is not None else None,
            "MACD Bullish (Yes/No)": ("Yes" if macd > macd_signal else "No") if (macd is not None and macd_signal is not None) else None,
            "Data Fetched": entry.fetch_time.strftime("%Y-%m-%d %H:%M:%S") if entry.fetch_time else None,
        })
    return pd.DataFrame(rows)


def run(input_path: str, output_path: str) -> None:
    log.info("Reading ticker list from %s", input_path)
    tickers = read_ticker_list(input_path, "Selected_Stocks")
    if not tickers:
        raise ValueError("No tickers found in 'Selected_Stocks'.")

    index_map = read_ticker_index_map(input_path, "Selected_Stocks")

    # 1. yfinance OHLCV
    yf_cache: Dict[str, TickerData] = {}
    log.info("Downloading yfinance data for %d tickers...", len(tickers))
    download_all(tickers, yf_cache)

    # 2. TradingView TA (with retry + delay)
    ta_cache: Dict[str, TickerTA] = {}
    log.info("Fetching TradingView TA data for %d tickers...", len(tickers))
    fetch_all_ta(tickers, ta_cache)

    # 3. FX rate
    log.info("Downloading USD/EGP exchange rate (%s)...", FX_TICKER)
    fx_series = download_fx(period=HISTORY_PERIOD)

    rows = []
    for raw in tickers:
        yf_entry = yf_cache.get(raw)
        if yf_entry is None or not yf_entry.ok:
            log.warning("Skipping %s: no yfinance data", raw)
            continue

        # Get TA data if available, else fallback to yfinance computations
        ta_entry = ta_cache.get(raw)
        if ta_entry and ta_entry.ok:
            ind = ta_entry.indicators
            sma50 = ind.get("SMA50")
            sma200 = ind.get("SMA200")
            ema20 = ind.get("EMA20")
            ema50 = ind.get("EMA50")
            ema200 = ind.get("EMA200")
            rsi = ind.get("RSI")
            close_ta = ind.get("close")
            vwma = ind.get("VWMA")  # Get VWMA from TA
            macd = ind.get("MACD.macd")
            macd_signal = ind.get("MACD.signal")
            pe_ratio = ind.get("price_earnings_ttm")
            adx = ind.get("ADX")
            adx_plus = ind.get("ADX+DI")
            adx_minus = ind.get("ADX-DI")
            bb_lower = ind.get("BB.lower")
            bb_upper = ind.get("BB.upper")
            ta_fetch_time = ta_entry.fetch_time.strftime("%Y-%m-%d %H:%M:%S") if ta_entry.fetch_time else None
        else:
            # Fallback: compute from yfinance
            log.info("%s: Using yfinance fallback for indicators", raw)
            fallback = compute_sma_ema_rsi_from_yf(yf_entry.history)
            sma50 = fallback["sma50"]
            sma200 = fallback["sma200"]
            ema20 = fallback["ema20"]
            ema50 = fallback["ema50"]
            ema200 = fallback["ema200"]
            rsi = fallback["rsi"]
            close_ta = None
            vwma = None  # No VWMA from yfinance fallback
            macd = fallback["macd"]
            macd_signal = fallback["macd_signal"]
            pe_ratio = None  # No P/E fallback source (yfinance history has no earnings data)
            adx = None
            adx_plus = None
            adx_minus = None
            bb_lower = None
            bb_upper = None
            ta_fetch_time = None
            ind = {}  # Empty indicators for fallback

        # Compute custom analytics from yfinance
        # MODIFIED: Pass ta_cache to usd_valuation for P/E-based fair value
        val = usd_valuation(raw, yf_cache, fx_series, ta_cache)
        vol = volume_analysis(raw, yf_cache, ta_cache)  # Pass ta_cache for volume
        mf = money_flow_volume_analysis(raw, yf_cache)
        sr = support_resistance(raw, yf_cache)

        # New indicators: ADL, MFI, BB Squeeze
        adl = compute_adl(yf_entry.history)
        mfi = compute_mfi(yf_entry.history)
        atr_val = None
        if "High" in yf_entry.history.columns and "Low" in yf_entry.history.columns:
            h = yf_entry.history["High"].values.astype(float)
            lo = yf_entry.history["Low"].values.astype(float)
            c = yf_entry.history["Close"].values.astype(float)
            if len(c) >= 2:
                trs = []
                for i in range(1, len(h)):
                    tr = max(h[i] - lo[i], abs(h[i] - c[i-1]), abs(lo[i] - c[i-1]))
                    trs.append(tr)
                atr_val = float(np.array(trs[-14:]).mean()) if trs else None
        bb_squeeze = compute_bb_squeeze(bb_lower, bb_upper, atr_val)

        # Prefer TradingView's close price - it's the freshest available price.
        # yfinance's OHLCV data can lag by up to a day depending on the data
        # provider, so it's kept only as a fallback when TA data is unavailable
        # for a ticker (valuation/undervaluation math below still uses
        # yfinance's own aligned history internally, since that needs to match
        # the FX series day-for-day).
        current_price = close_ta
        if current_price is None:
            current_price = val.get("current_egp")

        # Cross signals from SMA values (TA or fallback)
        golden_cross = None
        death_cross = None
        if sma50 is not None and sma200 is not None:
            if sma50 > sma200:
                golden_cross = "Yes"
                death_cross = "No"
            elif sma50 < sma200:
                golden_cross = "No"
                death_cross = "Yes"
            else:
                golden_cross = "No"
                death_cross = "No"

        ema_bullish = None
        if ema50 is not None and ema200 is not None:
            ema_bullish = "Yes" if ema50 > ema200 else "No"

        # Diamond Cross: EMA20 crossing above EMA50 (shorter-term momentum signal)
        diamond_cross = None
        if ema20 is not None and ema50 is not None:
            diamond_cross = "Yes" if ema20 > ema50 else "No"

        # MACD Bullish: MACD line above its signal line
        macd_bullish = None
        if macd is not None and macd_signal is not None:
            macd_bullish = "Yes" if macd > macd_signal else "No"

        # ChartScanAI: YOLOv8 candlestick pattern detection
        cs_result = chartscan_analyze(yf_entry.history, raw)
        cs_signal = cs_result["signal"] if cs_result else None
        cs_confidence = cs_result["confidence"] if cs_result else None
        cs_buy_patterns = cs_result["buy_patterns"] if cs_result else None
        cs_sell_patterns = cs_result["sell_patterns"] if cs_result else None

        # Enhanced Recommendation & Entry
        rec = build_enhanced_recommendation_and_entry(
            val, mf, sr, ind, current_price,
            sma50, sma200,
            golden_cross, death_cross,
            rsi,
            diamond_cross,
            adx, adx_plus, adx_minus,
            mfi, adl, bb_squeeze,
            cs_signal,
        )

        rows.append({
            "Analysis Run Date": datetime.now().strftime("%Y-%m-%d"),
            "Selected Stock": raw,
            "Index Membership": index_map.get(raw, "UNINDEX"),
            "Data As Of": yf_entry.history.index[-1].strftime("%Y-%m-%d"),
            "Current EGP Price": round(current_price, 4) if current_price is not None else None,
            "Current USD Price": round(val["current_usd"], 4) if val["current_usd"] is not None else None,
            "Historical Min USD Price": round(val["hist_min_usd"], 4) if val["hist_min_usd"] else None,
            "Historical Max USD Price": round(val["hist_max_usd"], 4) if val["hist_max_usd"] else None,
            "Undervalued (Yes/No)": val["undervalued"],
            "Implied Fair Value (EGP)": round(val["implied_fair_value_egp"], 4) if val["implied_fair_value_egp"] is not None else None,
            "Fair Value Method": val["fair_value_method"],
            "P/E Ratio (TTM)": round(val["pe_ratio_ttm"], 2) if val["pe_ratio_ttm"] is not None else None,
            "EPS (TTM)": round(val["eps_ttm"], 4) if val["eps_ttm"] is not None else None,
            "1-Year Avg Volume": round(vol["avg_vol_1y"], 0) if vol["avg_vol_1y"] else None,
            "Last Day Volume": round(vol["last_day_vol"], 0) if vol["last_day_vol"] else None,
            "Volume Multiplier (vs 1Y)": vol["vol_multiplier"],
            "Est. Buy Volume (2-Month Avg)": round(mf["buy_vol_avg_2mo"], 0) if mf["buy_vol_avg_2mo"] else None,
            "Est. Buy Volume (Last Day)": round(mf["buy_vol_last_day"], 0) if mf["buy_vol_last_day"] else None,
            "Buy Volume Multiplier (vs 2-Month)": mf["buy_vol_multiplier"],
            "Support": round(sr["support"], 4) if sr["support"] else None,
            "Resistance": round(sr["resistance"], 4) if sr["resistance"] else None,
            "50 SMA": round(sma50, 4) if sma50 is not None else None,
            "200 SMA": round(sma200, 4) if sma200 is not None else None,
            "Golden Cross (Yes/No)": golden_cross,
            "Death Cross (Yes/No)": death_cross,
            "20 EMA": round(ema20, 4) if ema20 is not None else None,
            "50 EMA": round(ema50, 4) if ema50 is not None else None,
            "200 EMA": round(ema200, 4) if ema200 is not None else None,
            "EMA Bullish (50>200) (Yes/No)": ema_bullish,
            "Diamond Cross (20>50) (Yes/No)": diamond_cross,
            "MACD": round(macd, 4) if macd is not None else None,
            "MACD Signal": round(macd_signal, 4) if macd_signal is not None else None,
            "MACD Bullish (Yes/No)": macd_bullish,
            "RSI (%)": round(rsi, 2) if rsi is not None else None,
            "VWMA": round(vwma, 4) if vwma is not None else None,
            "ADX": round(adx, 2) if adx is not None else None,
            "ADX +DI": round(adx_plus, 2) if adx_plus is not None else None,
            "ADX -DI": round(adx_minus, 2) if adx_minus is not None else None,
            "BB Lower": round(bb_lower, 4) if bb_lower is not None else None,
            "BB Upper": round(bb_upper, 4) if bb_upper is not None else None,
            "BB Squeeze": "Yes" if bb_squeeze else ("No" if bb_squeeze is not None else None),
            "ADL": round(adl, 0) if adl is not None else None,
            "MFI": round(mfi, 2) if mfi is not None else None,
            "TA Data As Of": ta_fetch_time,
            "Optimal Entry Price": rec["optimal_entry_price"],
            "Stop Loss": rec["stop_loss"],
            "Stop Loss Basis": rec["stop_loss_basis"],
            "Take Profit 1": rec["take_profit_1"],
            "Take Profit 2": rec["take_profit_2"],
            "Take Profit 3": rec["take_profit_3"],
            "Take Profit Basis": rec["take_profit_basis"],
            "TP1 Risk/Reward": rec["tp1_rr"],
            "TP2 Risk/Reward": rec["tp2_rr"],
            "TP3 Risk/Reward": rec["tp3_rr"],
            "TP1 Reward %": rec["tp1_reward_pct"],
            "TP2 Reward %": rec["tp2_reward_pct"],
            "TP3 Reward %": rec["tp3_reward_pct"],
            "Recommendation": rec["recommendation"],
            "Recommendation Basis": rec["recommendation_basis"],
            "ChartScanAI Signal": cs_signal,
            "ChartScanAI Confidence": round(cs_confidence, 4) if cs_confidence is not None else None,
            "ChartScanAI Buy Patterns": cs_buy_patterns,
            "ChartScanAI Sell Patterns": cs_sell_patterns,
        })

    if not rows:
        raise RuntimeError("No valid tickers to output; check logs.")

    out_df = pd.DataFrame(rows)

    log.info("Fetching EGX index snapshot (EGX30/EGX70/EGX33)...")
    index_df = fetch_index_snapshot()

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        out_df.to_excel(writer, sheet_name="Stock_Analysis", index=False)
        index_df.to_excel(writer, sheet_name="Indices", index=False)

    append_daily_history(out_df, output_path)

    log.info("Analysis complete. Output saved to %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="EGX Stock Analysis Tool (Hybrid with Enhanced Entry & TP)")
    parser.add_argument(
        "input_file",
        nargs="?",
        default=r"input.xlsx",
        help="Path to the input Excel workbook."
    )
    parser.add_argument(
        "-o", "--output",
        default=r"Stock_Analysis_Output.xlsx",
        help="Path to the output workbook."
    )
    args = parser.parse_args()

    try:
        run(args.input_file, args.output)
    except Exception as e:
        log.error("Fatal error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
