# -*- coding: utf-8 -*-
"""
EGX Stock Analysis Tool (Hybrid: TradingView TA + yfinance fallback)
====================================================================

Tries to fetch indicators from TradingView (with retries & delays).
If TA fails (e.g., 429 rate limit), falls back to computing from yfinance.

All original columns are preserved: USD valuation, volume, buy volume
multiplier (vs 2‑month avg), support/resistance, and recommendation.

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
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from tradingview_ta import TA_Handler, Interval

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

HISTORY_PERIOD = "10y"
MIN_TRADING_DAYS = 30
EGX_SUFFIX = ".CA"
AUTO_ADJUST = False
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
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70

VOLUME_SPIKE_MULTIPLIER = 1.5
NEAR_SUPPORT_PCT = 0.03
BUY_VOL_AVG_DAYS = 42          # 2‑month (trading days) for buy volume average

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

# TA fetch retry settings
TA_RETRIES = 5                  # attempts per ticker before giving up
TA_RETRY_DELAY = 2              # base seconds, doubles each retry (2,4,8,16,32)
TA_REQUEST_DELAY_MIN = 3        # min seconds between different tickers
TA_REQUEST_DELAY_MAX = 6        # max seconds between different tickers (randomized)

# Batch pause settings (pause longer every N tickers to respect rate limits)
TA_BATCH_SIZE = 12              # pause after this many tickers (10-15 range)
TA_BATCH_PAUSE_MIN = 45         # min seconds to pause between batches
TA_BATCH_PAUSE_MAX = 60         # max seconds to pause between batches

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

def fetch_ta(raw_ticker: str) -> TickerTA:
    """Single attempt to fetch TA data."""
    entry = TickerTA(raw_ticker=raw_ticker)
    try:
        handler = TA_Handler(
            symbol=raw_ticker,
            exchange="EGX",
            screener="egypt",
            interval=Interval.INTERVAL_1_DAY
        )
        analysis = handler.get_analysis()
        indicators = analysis.indicators

        if indicators.get("close") is None:
            entry.reason = "no close price"
            return entry

        entry.indicators = indicators
        entry.ok = True
        entry.fetch_time = datetime.now()  # Record fetch time
        return entry
    except Exception as e:
        entry.reason = f"TA fetch error: {e}"
        return entry


def fetch_ta_with_retry(raw_ticker: str) -> TickerTA:
    """Attempt to fetch TA with exponential backoff + jitter on 429 errors."""
    entry = None
    for attempt in range(TA_RETRIES):
        entry = fetch_ta(raw_ticker)
        if entry.ok:
            return entry
        # If it's a 429 (rate limit), wait (exponential backoff) and retry
        if "429" in entry.reason:
            if attempt == TA_RETRIES - 1:
                # last attempt already used, no point sleeping further
                break
            wait = TA_RETRY_DELAY * (2 ** attempt) + random.uniform(0, 1.5)
            log.warning("%s: 429 rate limit, retrying in %.1fs (attempt %d/%d)",
                        raw_ticker, wait, attempt + 1, TA_RETRIES)
            time.sleep(wait)
        else:
            # Non‑429 error, don't retry
            break
    return entry


def fetch_all_ta(tickers: list, cache: Dict[str, TickerTA]) -> None:
    """
    Fetch TA data for all tickers with:
      - randomized delay between each ticker (avoids fixed-interval pattern)
      - a longer pause every TA_BATCH_SIZE tickers to let any rate-limit window reset
    """
    for i, raw in enumerate(tickers):
        if raw in cache:
            continue
        entry = fetch_ta_with_retry(raw)
        cache[raw] = entry
        if entry.ok:
            log.info("%s: TA data loaded", raw)
        else:
            log.warning("%s: TA fetch failed - %s", raw, entry.reason)

        is_last = (i == len(tickers) - 1)
        if is_last:
            continue

        # Every TA_BATCH_SIZE tickers, take a longer pause to reset any rate-limit window
        if (i + 1) % TA_BATCH_SIZE == 0:
            batch_pause = random.uniform(TA_BATCH_PAUSE_MIN, TA_BATCH_PAUSE_MAX)
            log.info("Processed %d tickers, pausing %.1fs before next batch...",
                      i + 1, batch_pause)
            time.sleep(batch_pause)
        else:
            # Normal randomized delay between individual tickers
            time.sleep(random.uniform(TA_REQUEST_DELAY_MIN, TA_REQUEST_DELAY_MAX))


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
# Part 2: USD valuation analysis (EXACT copy from egx_stock_analysis_2.py)
# --------------------------------------------------------------------------

def usd_valuation(
    raw_ticker: str,
    cache: Dict[str, TickerData],
    fx_series: Optional[pd.Series],
) -> dict:
    """
    USD price for each day = EGP close / (EGP per USD) that day. Flags
    undervaluation by comparing today's USD price to the USD price on
    historical days when the stock traded at a SIMILAR EGP price (rather
    than comparing to its own all-time min/max, which would conflate
    equity moves with currency moves).
    """
    result = {
        "current_egp": None, "current_usd": None,
        "hist_min_usd": None, "hist_max_usd": None,
        "undervalued": "No",
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
    hist_min_usd = float(aligned["usd"].min())
    hist_max_usd = float(aligned["usd"].max())

    result.update(current_egp=current_egp, current_usd=current_usd,
                   hist_min_usd=hist_min_usd, hist_max_usd=hist_max_usd)

    lower = current_egp * (1 - EGP_SIMILARITY_BAND)
    upper = current_egp * (1 + EGP_SIMILARITY_BAND)
    comparable = aligned[(aligned["egp"] >= lower) & (aligned["egp"] <= upper)]
    comparable = comparable.iloc[:-2] if len(comparable) > 2 else comparable

    if len(comparable) >= 5:
        historical_median_usd = float(comparable["usd"].median())
        if current_usd < historical_median_usd:
            result["undervalued"] = "Yes"

    return result


# --------------------------------------------------------------------------
# Volume analysis (raw volume, 1‑year avg) - MODIFIED to use TA volume
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
# Money Flow Volume (buy volume) – 2‑month average
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
    result = {"sma50": None, "sma200": None, "ema20": None, "ema50": None, "ema200": None, "rsi": None}

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
) -> dict:
    """
    Enhanced recommendation with better entry price using TradingView indicators.
    Falls back to original logic if USE_ENHANCED_ENTRY is False or if TA data is missing.
    """
    reasons = []

    # --- Existing signals (unchanged) ---
    is_undervalued = val.get("undervalued") == "Yes"
    if is_undervalued:
        reasons.append("undervalued")

    buy_multiplier = mf.get("buy_vol_multiplier")
    is_volume_spike = buy_multiplier is not None and buy_multiplier >= VOLUME_SPIKE_MULTIPLIER
    if is_volume_spike:
        reasons.append("buy-volume spike (vs 2‑month)")

    support = sr.get("support")
    is_near_support = False
    if current_price is not None and support is not None and support > 0:
        distance_pct = (current_price - support) / support
        is_near_support = 0 <= distance_pct <= NEAR_SUPPORT_PCT
    if is_near_support:
        reasons.append("near support")

    signal_count = sum([is_undervalued, is_volume_spike, is_near_support])

    # --- Determine trend regime ---
    if golden_cross == "Yes":
        regime = "bullish"
    elif death_cross == "Yes":
        regime = "bearish"
    else:
        regime = "unknown"

    # --- Enhanced Entry Price ---
    if USE_ENHANCED_ENTRY and ind and len(ind) > 0:
        # Use enhanced entry logic with TradingView indicators
        entry_price, entry_basis = get_enhanced_entry_price(
            ind,
            sr.get("support"),
            current_price,
            regime,
            is_near_support
        )
    else:
        # Fallback to original logic
        entry_price, entry_basis = None, ""
        if regime == "bullish":
            if is_near_support and support is not None:
                entry_price = support
                entry_basis = "pullback entry at swing‑low support in a confirmed uptrend"
            elif sma_50 is not None and current_price is not None and sma_50 < current_price:
                entry_price = sma_50
                entry_basis = "pullback entry at the rising 50 SMA in a confirmed uptrend"
            else:
                entry_price = current_price
                entry_basis = "no pullback level identified; current price as reference"
        elif regime == "bearish":
            entry_price = None
            entry_basis = "confirmed downtrend (Death Cross); no technical entry recommended"
        else:
            entry_price = None
            entry_basis = "trend undetermined (insufficient 200‑period history)"

    # --- Enhanced Stop Loss ---
    if USE_ENHANCED_ENTRY and entry_price is not None and ind and len(ind) > 0:
        stop_loss, stop_loss_basis = get_enhanced_stop_loss(
            ind,
            sr.get("support"),
            entry_price,
            current_price
        )
    else:
        # Fallback: 5% below entry
        if entry_price is not None:
            stop_loss = entry_price * 0.95
            stop_loss_basis = "stop at 5% below entry (default)"
        else:
            stop_loss = None
            stop_loss_basis = "no stop loss (no entry)"

    # --- Take Profit Levels ---
    if USE_ENHANCED_ENTRY and entry_price is not None and ind and len(ind) > 0:
        tp_results = calculate_take_profit_levels(ind, current_price, entry_price)
        tp1 = tp_results["take_profit_1"]
        tp2 = tp_results["take_profit_2"]
        tp3 = tp_results["take_profit_3"]
        tp_basis = tp_results["take_profit_basis"]
    else:
        tp1 = tp2 = tp3 = None
        tp_basis = "no take profit levels (insufficient TA data)"

    # --- Risk/Reward Ratios ---
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

    # --- Recommendation (unchanged logic) ---
    if regime == "bearish":
        recommendation = "Avoid"
        reasons.append("Death Cross (confirmed downtrend)")
    elif regime == "bullish":
        overbought = rsi is not None and rsi >= RSI_OVERBOUGHT
        reasons.append("Golden Cross confirmed")
        has_diamond = diamond_cross == "Yes"
        if has_diamond:
            reasons.append("Diamond Cross (EMA20>EMA50) - short-term momentum bullish")
        if overbought:
            recommendation = "Watch"
            reasons.append(f"RSI overbought ({rsi:.1f})")
        elif signal_count == 3:
            recommendation = "Buy"
        elif signal_count == 2:
            # Diamond Cross adds extra momentum confluence -> upgrade to Buy
            recommendation = "Buy" if has_diamond else "Watch"
        elif signal_count <= 1 and has_diamond:
            # Fresh momentum shift even without other signals is worth a closer look
            recommendation = "Watch"
        else:
            recommendation = "Hold"
    else:  # unknown
        reasons.append("trend undetermined")
        if signal_count == 3:
            recommendation = "Buy"
        elif signal_count == 2:
            recommendation = "Watch"
        else:
            recommendation = "Hold"

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
# Main orchestration
# --------------------------------------------------------------------------

def run(input_path: str, output_path: str) -> None:
    log.info("Reading ticker list from %s", input_path)
    tickers = read_ticker_list(input_path, "Selected_Stocks")
    if not tickers:
        raise ValueError("No tickers found in 'Selected_Stocks'.")

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
            ta_fetch_time = None
            ind = {}  # Empty indicators for fallback

        # Compute custom analytics from yfinance
        val = usd_valuation(raw, yf_cache, fx_series)
        vol = volume_analysis(raw, yf_cache, ta_cache)  # Pass ta_cache for volume
        mf = money_flow_volume_analysis(raw, yf_cache)
        sr = support_resistance(raw, yf_cache)

        current_price = val.get("current_egp")  # from yfinance; fallback to TA close
        if current_price is None:
            current_price = close_ta

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

        # Enhanced Recommendation & Entry
        rec = build_enhanced_recommendation_and_entry(
            val, mf, sr, ind, current_price,
            sma50, sma200,
            golden_cross, death_cross,
            rsi,
            diamond_cross,
        )

        rows.append({
            "Selected Stock": raw,
            "Data As Of": yf_entry.history.index[-1].strftime("%Y-%m-%d"),
            "Current EGP Price": round(current_price, 4) if current_price is not None else None,
            "Current USD Price": round(val["current_usd"], 4) if val["current_usd"] is not None else None,
            "Historical Min USD Price": round(val["hist_min_usd"], 4) if val["hist_min_usd"] else None,
            "Historical Max USD Price": round(val["hist_max_usd"], 4) if val["hist_max_usd"] else None,
            "Undervalued (Yes/No)": val["undervalued"],
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
            "RSI (%)": round(rsi, 2) if rsi is not None else None,
            "VWMA": round(vwma, 4) if vwma is not None else None,  # New column for VWMA
            "TA Data As Of": ta_fetch_time,  # New column for TA fetch time
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
        })

    if not rows:
        raise RuntimeError("No valid tickers to output; check logs.")

    out_df = pd.DataFrame(rows)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        out_df.to_excel(writer, sheet_name="Stock_Analysis", index=False)

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