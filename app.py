"""
app.py - Sahi Key Levels LIVE (cross-timeframe validated zones + alerts)

Builds on the sahi-key-levels module (vendored below, UNCHANGED) to add:
  1. Two independently-computed zone sets per symbol: a COMPOSITE profile
     (18 trading days) and an INTRADAY profile (today's session only).
  2. Cross-timeframe validation (see zone_validation.py): a zone only
     counts as a real level if its price range shows up in BOTH profiles.
     A zone with no multi-day backing, or a composite zone today's session
     hasn't touched at all, is treated as noise and dropped.
  3. BUY/SELL alerts, edge-triggered (only logged the moment a symbol's
     signal changes, not every refresh) and gated to market hours (same
     fix already applied in hvn-lvn-scanner: an after-hours refresh pulls
     Upstox's frozen post-close quotes, which must not get logged as a
     live signal).
  4. A Chart tab: candlesticks for today's session with composite,
     intraday, and validated zones overlaid (see candles_with_levels.py).

VENDORED FILES (copied unchanged from their source repos, per instruction
to leave the original logic untouched):
    hvn_lvn.py               <- from hvn-lvn-scanner
    sahi_style_key_levels.py <- from sahi-key-levels

THREE-TIER REFRESH MODEL (deliberate, not accidental complexity):
  - "Run Precompute" (slow, once/day): resolves instrument keys, fetches
    18 days of 5-min candles per symbol, computes the COMPOSITE zone set.
    This is the expensive step -- same reasoning as hvn-lvn-scanner's
    Precompute.
  - "Refresh Zones" (medium, every few minutes -- NOT on every quote tick):
    re-fetches ONLY today's 5-min candles per symbol (a much lighter
    historical-candle call than the 18-day Precompute fetch, but still one
    HTTP call per symbol, so this is not free -- don't wire it to run on
    every quote refresh across 200+ symbols). Recomputes the INTRADAY zone
    set, cross-validates against the cached COMPOSITE set, and logs any
    new BUY/SELL alerts.
  - "Refresh Quotes" (fast): single batch quote call for LTP/VWAP, same as
    hvn-lvn-scanner's existing fast refresh. Recomputes each symbol's
    signal against whatever zones were last computed by "Refresh Zones"
    (may be a few minutes stale) -- this keeps the Scanner table feeling
    responsive without re-fetching candles on every tick.

SETUP:
    pip install streamlit requests pandas numpy plotly --break-system-packages
    $env:UPSTOX_ACCESS_TOKEN = "your_token_here"
    streamlit run app.py
"""
import os
import re
import json
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone, time as dtime

import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from hvn_lvn import build_volume_profile, find_hvn_lvn
from sahi_style_key_levels import sahi_style_key_levels
from zone_validation import cross_validated_zones, compute_zone_signal
from candles_with_levels import plot_candles_with_zones
from live_feed_reader import get_live_candles

IST = timezone(timedelta(hours=5, minutes=30))

def now_ist():
    return datetime.now(IST)

# ---------------- Config ----------------
INSTRUMENT_SEARCH_URL = "https://api.upstox.com/v2/instruments/search"
QUOTES_URL = "https://api.upstox.com/v2/market-quote/quotes"
CACHE_PATH = "sahi_zones_cache.json"
ALERT_LOG_PATH = "alert_log.json"

DAILY_LOOKBACK_DAYS = 30         # needs enough history for RVOL_BASELINE_DAYS average
COMPOSITE_LOOKBACK_DAYS = 18     # matches hvn-lvn-scanner's multi-day window
RVOL_BASELINE_DAYS = 20          # prior-N-day average full-day volume, same convention as hvn-lvn-scanner
TOP_N_RVOL = 5                   # only symbols in the top N by RVOL are eligible to alert

COMPOSITE_N_BINS = 50
INTRADAY_N_BINS = 45
MIN_PROMINENCE_PCT = 0.08
MIN_BIN_DISTANCE = 2
MAX_ZONES = 6
MIN_DISPLAY_PCT = 2.0
MIN_SIGNAL_DISTANCE_PCT = 0.5    # how far LTP must be from a validated zone to signal
MIN_VWAP_DISTANCE_PCT = 0.15     # how far LTP must be from VWAP before a bias counts as real
                                  # (found necessary live: without this, tiny VWAP wobbles of
                                  # 0.02-0.05% fired repeated BUY/SELL flips on the same symbol)
AUTO_REFRESH_QUOTES_SECONDS = 60     # quotes + signal recompute cadence when auto-refresh is on
ZONE_REFRESH_EVERY_N_TICKS = 5       # also do a heavier zone refresh every Nth tick (~5 min)

MARKET_OPEN_TIME = dtime(9, 15)   # IST - no new alerts logged before this
MARKET_CLOSE_TIME = dtime(15, 30)  # IST - no new alerts logged at/after this

NEAR_ZONE_PCT = 0.3   # how close (%) LTP must be to a validated zone edge to count as "at" it

# Full liquid NSE F&O universe (not restricted to Nifty 50 anymore) --
# same universe proven out across the other repos (hvn-lvn-scanner,
# fno-scanner-strategy-update). NIFTY/BANKNIFTY handled separately below
# as futures, not equity. This list drifts over time as NSE adds/removes
# F&O eligibility, so it's worth a periodic sanity check, not treated as
# permanently fixed.
EQUITY_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "SBIN", "AXISBANK",
    "KOTAKBANK", "BAJFINANCE", "BHARTIARTL", "ITC", "LT", "HINDUNILVR",
    "MARUTI", "TMPV", "TATASTEEL", "SUNPHARMA", "TITAN", "ULTRACEMCO",
    "ASIANPAINT", "WIPRO", "NTPC", "POWERGRID", "M&M", "ADANIENT",
    "ADANIPORTS", "BAJAJFINSV", "HCLTECH", "JSWSTEEL", "ONGC", "COALINDIA",
    "TECHM", "GRASIM", "DIVISLAB", "DRREDDY", "CIPLA", "EICHERMOT",
    "HEROMOTOCO", "HINDALCO", "BPCL", "BRITANNIA", "APOLLOHOSP", "SBILIFE",
    "HDFCLIFE", "INDUSINDBK", "BAJAJ-AUTO", "TATACONSUM", "UPL", "SHREECEM",
    "NESTLEIND", "VEDANTA", "GAIL", "PIDILITIND", "DLF", "GODREJCP",
    "SIEMENS", "AMBUJACEM", "BANDHANBNK", "BANKBARODA", "PNB", "CANBK",
    "IDFCFIRSTB", "FEDERALBNK", "AUROPHARMA", "BEL", "BIOCON", "CHOLAFIN",
    "COLPAL", "CONCOR", "CUMMINSIND", "DABUR", "DEEPAKNTR", "ESCORTS",
    "EXIDEIND", "GODREJPROP", "HAVELLS", "HDFCAMC", "ICICIGI", "ICICIPRULI",
    "IEX", "INDIGO", "INDUSTOWER", "IOC", "IRCTC", "JINDALSTEL", "JUBLFOOD",
    "LICHSGFIN", "LTIM", "LUPIN", "MANAPPURAM", "MARICO", "MCDOWELL-N",
    "MFSL", "MOTHERSON", "MPHASIS", "MRF", "MUTHOOTFIN", "NAUKRI",
    "NMDC", "OBEROIRLTY", "OFSS", "PAGEIND", "PEL", "PERSISTENT",
    "PETRONET", "PFC", "PIIND", "POLYCAB", "RECLTD", "SAIL", "SBICARD",
    "SRF", "SYNGENE", "TATACOMM", "TATAPOWER", "TORNTPHARM", "TRENT",
    "TVSMOTOR", "UBL", "VOLTAS", "ZEEL", "ZYDUSLIFE", "CDSL", "IRFC",
    "IDEA", "YESBANK", "SUZLON", "ETERNAL", "DMART", "JIOFIN", "PAYTM",
    "NYKAA", "POLICYBZR", "DELHIVERY", "LODHA", "PATANJALI", "ABCAPITAL",
    "ALKEM", "APLAPOLLO", "ASHOKLEY", "ASTRAL", "ATUL", "BALKRISNIND",
    "BATAINDIA", "BHARATFORG", "BHEL", "BSOFT", "CANFINHOME", "CROMPTON",
    "CUB", "DALBHARAT", "GLENMARK", "GMRINFRA", "GNFC", "GRANULES",
    "GUJGASLTD", "HAL", "HINDCOPPER", "HINDPETRO", "IBULHSGFIN", "IGL",
    "INDHOTEL", "INDIAMART", "IPCALAB", "JKCEMENT", "L&TFH", "LALPATHLAB",
    "LAURUSLABS", "M&MFIN", "METROPOLIS", "NATIONALUM", "NAVINFLUOR",
    "OIL", "PVRINOX", "RAIN", "RBLBANK", "SUNTV", "TATACHEM",
    "TATAELXSI", "TORNTPOWER", "UNIONBANK", "VBL", "WHIRLPOOL",
    "AARTIIND", "ABFRL", "ANGELONE", "APOLLOTYRE", "AUBANK", "BANKINDIA",
    "BSE", "CGPOWER", "CHAMBLFERT", "COFORGE", "COROMANDEL", "DIXON",
    "FORTIS", "GICRE", "GODFRYPHLP", "GRAPHITE", "GSPL", "HFCL",
    "HUDCO", "IIFL", "INDIACEM", "IRB", "ITI", "KALYANKJIL",
    "KEI", "LTF", "MANKIND", "MAXHEALTH", "MGL", "MOTILALOFS",
    "NBCC", "NCC", "NHPC", "PFIZER", "PGEL", "POWERINDIA",
    "PRESTIGE", "RVNL", "SJVN", "SOLARINDS", "SONACOMS", "STARHEALTH",
    "SUPREMEIND", "TIINDIA", "TITAGARH", "VEDL", "ZFCVINDIA",
    "SHRIRAMFIN",
]
FUTURES_SYMBOLS = ["NIFTY", "BANKNIFTY"]

# Sector grouping for the full F&O universe above -- used by the Sectors
# tab to render a grid of small charts for one sector at a time. Best-
# effort NSE-style categorization; a few names are genuinely borderline
# (e.g. Adani Enterprises is diversified, PFC/RECLTD are PSU financiers
# grouped under NBFC here) -- treat this as a practical scanning grouping,
# not a formal index classification.
SECTOR_MAP = {
    # Banks
    "HDFCBANK": "Banks", "ICICIBANK": "Banks", "SBIN": "Banks",
    "KOTAKBANK": "Banks", "AXISBANK": "Banks", "INDUSINDBK": "Banks",
    "BANDHANBNK": "Banks", "BANKBARODA": "Banks", "PNB": "Banks",
    "CANBK": "Banks", "IDFCFIRSTB": "Banks", "FEDERALBNK": "Banks",
    "RBLBANK": "Banks", "AUBANK": "Banks", "BANKINDIA": "Banks",
    "UNIONBANK": "Banks", "CUB": "Banks", "YESBANK": "Banks",

    # NBFC / Financial Services
    "BAJFINANCE": "NBFC", "BAJAJFINSV": "NBFC", "CHOLAFIN": "NBFC",
    "MANAPPURAM": "NBFC", "MUTHOOTFIN": "NBFC", "LICHSGFIN": "NBFC",
    "MFSL": "NBFC", "PFC": "NBFC", "RECLTD": "NBFC", "SBICARD": "NBFC",
    "ABCAPITAL": "NBFC", "CANFINHOME": "NBFC", "IBULHSGFIN": "NBFC",
    "L&TFH": "NBFC", "M&MFIN": "NBFC", "LTF": "NBFC", "HUDCO": "NBFC",
    "IIFL": "NBFC", "MOTILALOFS": "NBFC", "ANGELONE": "NBFC",
    "JIOFIN": "NBFC", "SHRIRAMFIN": "NBFC", "HDFCAMC": "NBFC",
    "PAYTM": "NBFC", "POLICYBZR": "NBFC", "PEL": "NBFC",

    # Insurance
    "SBILIFE": "Insurance", "HDFCLIFE": "Insurance", "ICICIGI": "Insurance",
    "ICICIPRULI": "Insurance", "STARHEALTH": "Insurance", "GICRE": "Insurance",

    # Financial Infra / Exchanges
    "BSE": "Financial Infra", "CDSL": "Financial Infra", "IEX": "Financial Infra",

    # IT
    "TCS": "IT", "INFY": "IT", "HCLTECH": "IT", "WIPRO": "IT", "TECHM": "IT",
    "LTIM": "IT", "MPHASIS": "IT", "PERSISTENT": "IT", "COFORGE": "IT",
    "OFSS": "IT", "NAUKRI": "IT", "BSOFT": "IT", "TATAELXSI": "IT",
    "INDIAMART": "IT",

    # Auto & Ancillaries
    "MARUTI": "Auto", "M&M": "Auto", "TMPV": "Auto", "EICHERMOT": "Auto",
    "HEROMOTOCO": "Auto", "BAJAJ-AUTO": "Auto", "TVSMOTOR": "Auto",
    "ASHOKLEY": "Auto", "MOTHERSON": "Auto", "BHARATFORG": "Auto",
    "BALKRISNIND": "Auto", "MRF": "Auto", "APOLLOTYRE": "Auto",
    "EXIDEIND": "Auto", "ESCORTS": "Auto", "SONACOMS": "Auto",
    "TIINDIA": "Auto", "ZFCVINDIA": "Auto",

    # Pharma & Healthcare
    "SUNPHARMA": "Pharma", "DRREDDY": "Pharma", "CIPLA": "Pharma",
    "DIVISLAB": "Pharma", "AUROPHARMA": "Pharma", "BIOCON": "Pharma",
    "LUPIN": "Pharma", "TORNTPHARM": "Pharma", "ALKEM": "Pharma",
    "GLENMARK": "Pharma", "GRANULES": "Pharma", "LAURUSLABS": "Pharma",
    "IPCALAB": "Pharma", "ZYDUSLIFE": "Pharma", "MANKIND": "Pharma",
    "SYNGENE": "Pharma", "PFIZER": "Pharma",
    "APOLLOHOSP": "Healthcare", "FORTIS": "Healthcare", "MAXHEALTH": "Healthcare",
    "LALPATHLAB": "Healthcare", "METROPOLIS": "Healthcare",

    # FMCG
    "HINDUNILVR": "FMCG", "ITC": "FMCG", "TATACONSUM": "FMCG",
    "BRITANNIA": "FMCG", "NESTLEIND": "FMCG", "DABUR": "FMCG",
    "MARICO": "FMCG", "COLPAL": "FMCG", "GODREJCP": "FMCG",
    "MCDOWELL-N": "FMCG", "UBL": "FMCG", "VBL": "FMCG", "PATANJALI": "FMCG",
    "JUBLFOOD": "FMCG", "GODFRYPHLP": "FMCG",

    # Metals & Mining
    "TATASTEEL": "Metals & Mining", "JSWSTEEL": "Metals & Mining",
    "HINDALCO": "Metals & Mining", "ADANIENT": "Metals & Mining",
    "VEDANTA": "Metals & Mining", "VEDL": "Metals & Mining",
    "NMDC": "Metals & Mining", "NATIONALUM": "Metals & Mining",
    "HINDCOPPER": "Metals & Mining", "SAIL": "Metals & Mining",
    "JINDALSTEL": "Metals & Mining", "APLAPOLLO": "Metals & Mining",

    # Oil & Gas / Energy
    "RELIANCE": "Oil & Gas", "ONGC": "Oil & Gas", "COALINDIA": "Oil & Gas",
    "GAIL": "Oil & Gas", "BPCL": "Oil & Gas", "IOC": "Oil & Gas",
    "HINDPETRO": "Oil & Gas", "PETRONET": "Oil & Gas", "OIL": "Oil & Gas",
    "GSPL": "Oil & Gas", "IGL": "Oil & Gas", "MGL": "Oil & Gas",
    "GUJGASLTD": "Oil & Gas",

    # Power
    "NTPC": "Power", "POWERGRID": "Power", "TATAPOWER": "Power",
    "TORNTPOWER": "Power", "NHPC": "Power", "SJVN": "Power",
    "CGPOWER": "Power",

    # Capital Goods & Defence
    "LT": "Capital Goods", "SIEMENS": "Capital Goods", "CUMMINSIND": "Capital Goods",
    "BHEL": "Capital Goods", "BEL": "Capital Goods", "HAL": "Capital Goods",
    "POLYCAB": "Capital Goods", "KEI": "Capital Goods", "SOLARINDS": "Capital Goods",
    "POWERINDIA": "Capital Goods", "GRAPHITE": "Capital Goods",
    "SUZLON": "Capital Goods", "ITI": "Capital Goods",

    # Cement & Construction Materials
    "ULTRACEMCO": "Cement", "GRASIM": "Cement", "SHREECEM": "Cement",
    "AMBUJACEM": "Cement", "DALBHARAT": "Cement", "JKCEMENT": "Cement",
    "INDIACEM": "Cement",

    # Chemicals
    "PIDILITIND": "Chemicals", "UPL": "Chemicals", "SRF": "Chemicals",
    "DEEPAKNTR": "Chemicals", "ATUL": "Chemicals", "GNFC": "Chemicals",
    "NAVINFLUOR": "Chemicals", "AARTIIND": "Chemicals", "TATACHEM": "Chemicals",
    "PIIND": "Chemicals", "ASTRAL": "Chemicals", "RAIN": "Chemicals",
    "SUPREMEIND": "Chemicals",

    # Consumer Durables
    "TITAN": "Consumer Durables", "ASIANPAINT": "Consumer Durables",
    "HAVELLS": "Consumer Durables", "VOLTAS": "Consumer Durables",
    "CROMPTON": "Consumer Durables", "DIXON": "Consumer Durables",
    "WHIRLPOOL": "Consumer Durables", "BATAINDIA": "Consumer Durables",
    "PGEL": "Consumer Durables",

    # Telecom
    "BHARTIARTL": "Telecom", "INDUSTOWER": "Telecom", "IDEA": "Telecom",
    "HFCL": "Telecom", "TATACOMM": "Telecom",

    # Realty
    "DLF": "Realty", "GODREJPROP": "Realty", "OBEROIRLTY": "Realty",
    "PRESTIGE": "Realty", "LODHA": "Realty",

    # Media & Entertainment
    "ZEEL": "Media", "SUNTV": "Media", "PVRINOX": "Media",

    # Retail / Consumer Services
    "TRENT": "Retail", "ETERNAL": "Retail", "DMART": "Retail",
    "NYKAA": "Retail", "PAGEIND": "Retail", "ABFRL": "Retail",
    "KALYANKJIL": "Retail",

    # Aviation / Logistics
    "INDIGO": "Aviation & Logistics", "CONCOR": "Aviation & Logistics",
    "GMRINFRA": "Aviation & Logistics", "DELHIVERY": "Aviation & Logistics",

    # Hotels & Travel
    "IRCTC": "Hotels & Travel", "INDHOTEL": "Hotels & Travel",

    # Construction & Infra
    "IRB": "Construction & Infra", "NBCC": "Construction & Infra",
    "NCC": "Construction & Infra", "RVNL": "Construction & Infra",
    "TITAGARH": "Construction & Infra",

    # Agri & Fertilizers
    "CHAMBLFERT": "Agri & Fertilizers", "COROMANDEL": "Agri & Fertilizers",

    # PSU Financial (rail/infra financing, distinct enough from private NBFC)
    "IRFC": "PSU Financial",

    # Diversified / Services
    "ADANIPORTS": "Diversified / Services",
}


def get_token():
    token = os.environ.get("UPSTOX_ACCESS_TOKEN")
    if token:
        return token.strip()
    try:
        if "UPSTOX_ACCESS_TOKEN" in st.secrets:
            return st.secrets["UPSTOX_ACCESS_TOKEN"].strip()
    except Exception:
        pass
    if os.path.exists("upstox_token.txt"):
        with open("upstox_token.txt", "r") as f:
            t = f.read().strip()
        if t and t != "PASTE_YOUR_TOKEN_HERE":
            return t
    raise RuntimeError(
        "No token found. Set $env:UPSTOX_ACCESS_TOKEN, add UPSTOX_ACCESS_TOKEN to Streamlit "
        "secrets, or create upstox_token.txt."
    )


def resolve_equity_instrument_key(symbol, token):
    headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
    params = {"query": symbol, "exchanges": "NSE", "segments": "EQ",
              "instrument_types": "EQ", "page_number": 1, "records": 10}
    resp = requests.get(INSTRUMENT_SEARCH_URL, headers=headers, params=params, timeout=20)
    resp.raise_for_status()
    candidates = [inst for inst in resp.json().get("data", [])
                  if inst.get("trading_symbol", "").upper() == symbol.upper()]
    return candidates[0]["instrument_key"] if candidates else None


def resolve_futures_instrument_key(name, token):
    """No expiry filter - sorts client-side by expiry (same fix already
    applied in hvn-lvn-scanner: 'current_month' keyword returns zero
    results once that month's contract expires but the calendar hasn't
    rolled over yet)."""
    headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
    params = {"query": name, "exchanges": "NSE", "segments": "FO",
              "instrument_types": "FUT", "page_number": 1, "records": 30}
    resp = requests.get(INSTRUMENT_SEARCH_URL, headers=headers, params=params, timeout=20)
    resp.raise_for_status()
    candidates = [inst for inst in resp.json().get("data", [])
                  if inst.get("instrument_type") == "FUT"
                  and inst.get("underlying_symbol", "").upper() == name.upper()]
    if not candidates:
        return None
    candidates.sort(key=lambda x: x["expiry"])
    return candidates[0]["instrument_key"]


def fetch_candles(instrument_key, token, unit, interval, lookback_days):
    to_date = now_ist().strftime("%Y-%m-%d")
    from_date = (now_ist() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    url = f"https://api.upstox.com/v3/historical-candle/{instrument_key}/{unit}/{interval}/{to_date}/{from_date}"
    headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    candles = resp.json().get("data", {}).get("candles", [])
    if not candles:
        return pd.DataFrame()
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["date"] = df["timestamp"].dt.date
    return df


def fetch_5min_candles_ending(instrument_key, token, end_date, total_days):
    """Same as fetch_candles(unit='minutes', interval='5', ...) but anchored
    to an arbitrary past end_date instead of 'now' -- needed for the Replay
    tab, which looks at historical sessions, not today.

    Chunks into <=20-day windows and concatenates: Upstox's 5-min
    historical-candle endpoint rejects overly wide date ranges in one call
    (a 400 for ~30+ days -- same limit discovered and worked around in
    backtest_zone_formation.py)."""
    all_chunks = []
    remaining = total_days
    cursor_end = end_date
    headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}

    while remaining > 0:
        chunk_days = min(20, remaining)
        chunk_start = cursor_end - timedelta(days=chunk_days)
        url = (f"https://api.upstox.com/v3/historical-candle/{instrument_key}/minutes/5/"
               f"{cursor_end.strftime('%Y-%m-%d')}/{chunk_start.strftime('%Y-%m-%d')}")
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        candles = resp.json().get("data", {}).get("candles", [])
        if candles:
            all_chunks.append(pd.DataFrame(
                candles, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"]
            ))
        cursor_end = chunk_start
        remaining -= chunk_days

    if not all_chunks:
        return pd.DataFrame()

    df = pd.concat(all_chunks, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    df["date"] = df["timestamp"].dt.date
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_replay_data(instrument_key, token, replay_date_str):
    """Cached per (symbol, date) -- the Replay tab's slider re-runs this
    function's CALLER on every drag, but the actual fetch only happens
    once per symbol/date pick, not once per slider position. Returns
    (composite_zones, day_df) or (None, None) if there's not enough
    history or no candles for that day."""
    replay_date = datetime.strptime(replay_date_str, "%Y-%m-%d").date()
    total_days = COMPOSITE_LOOKBACK_DAYS + 25  # buffer for weekends/holidays
    full_df = fetch_5min_candles_ending(instrument_key, token, replay_date, total_days)
    if full_df.empty:
        return None, None

    trading_days = sorted(full_df["date"].unique())
    if replay_date not in trading_days:
        return None, None
    day_idx = trading_days.index(replay_date)
    composite_days = trading_days[max(0, day_idx - COMPOSITE_LOOKBACK_DAYS):day_idx]
    if not composite_days:
        return None, None

    composite_df = full_df[full_df["date"].isin(composite_days)]
    composite_zones = compute_composite_zones(composite_df)
    day_df = full_df[full_df["date"] == replay_date].reset_index(drop=True)
    return composite_zones, day_df


def compute_composite_zones(intraday_df):
    """Composite zone set from the FULL multi-day intraday_df (no date
    filtering -- composite means across all fetched days)."""
    if intraday_df.empty:
        return []
    try:
        _, shown = sahi_style_key_levels(
            intraday_df, n_bins=COMPOSITE_N_BINS, max_zones=MAX_ZONES,
            min_display_pct=MIN_DISPLAY_PCT, min_prominence_pct=MIN_PROMINENCE_PCT,
            min_bin_distance=MIN_BIN_DISTANCE,
        )
        return [asdict(z) for z in shown]
    except Exception:
        return []


def compute_intraday_zones(today_only_df):
    """Intraday zone set from a candle df already scoped to a single
    session (see fetch_today_candles below)."""
    if today_only_df.empty:
        return []
    today = today_only_df["date"].max()
    today_df = today_only_df[today_only_df["date"] == today]
    try:
        _, shown = sahi_style_key_levels(
            today_df, n_bins=INTRADAY_N_BINS, max_zones=MAX_ZONES,
            min_display_pct=MIN_DISPLAY_PCT, min_prominence_pct=MIN_PROMINENCE_PCT,
            min_bin_distance=MIN_BIN_DISTANCE,
        )
        return [asdict(z) for z in shown]
    except Exception:
        return []


def fetch_intraday_candles(instrument_key, token, unit="minutes", interval="5"):
    """Upstox's historical-candle endpoint (fetch_candles above) NEVER
    includes the still-open trading day -- it only has data up through
    yesterday's final close. Today's still-forming candles require this
    separate intraday endpoint. Without this, "today's" fetch silently
    returns only yesterday's last candle, which looks like a frozen/stale
    chart rather than an obvious error."""
    url = f"https://api.upstox.com/v3/historical-candle/intraday/{instrument_key}/{unit}/{interval}"
    headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    candles = resp.json().get("data", {}).get("candles", [])
    if not candles:
        return pd.DataFrame()
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["date"] = df["timestamp"].dt.date
    return df


def fetch_today_candles(instrument_key, token):
    """Today's session candles via the intraday endpoint. Falls back to
    the historical endpoint's last available day (e.g. before market
    open, when the intraday endpoint may return nothing yet) so the
    Chart/zone functions always get something to work with."""
    df = fetch_intraday_candles(instrument_key, token, "minutes", "5")
    if not df.empty:
        return df

    df = fetch_candles(instrument_key, token, "minutes", "5", lookback_days=1)
    if df.empty:
        return df
    latest = df["date"].max()
    return df[df["date"] == latest]


@st.cache_data(ttl=45, show_spinner=False)
def fetch_today_candles_cached(instrument_key, token):
    """Same as fetch_today_candles, but memoized for 45s. The all-sectors
    scroll view renders 200+ small charts at once -- without this, every
    script rerun (including auto-refresh ticks) would re-fetch candles
    for all 200+ symbols, which is slow and hammers the Upstox API far
    harder than necessary for a view that's mostly just being scrolled,
    not actively refreshed every few seconds."""
    return fetch_today_candles(instrument_key, token)


def get_today_candles(symbol, instrument_key, token):
    """Tries the live WebSocket feed first (fno-websocket-feed's shared
    JSON file) -- instant, no API call, genuinely live. Falls back to the
    existing REST fetch if the feed isn't running, is stale, or doesn't
    have this symbol yet (e.g. it's an index key the feed can't name, or
    the listener only just started). This fallback is what keeps
    Streamlit Cloud working exactly as before -- the live feed file will
    simply never exist there, so every call just uses REST, unchanged."""
    live_df = get_live_candles(symbol)
    if live_df is not None and not live_df.empty:
        return live_df
    return fetch_today_candles_cached(instrument_key, token)


def run_precompute(token, progress_callback=None):
    cache = {}
    all_symbols = [(s, "equity") for s in EQUITY_SYMBOLS] + [(s, "futures") for s in FUTURES_SYMBOLS]
    for i, (symbol, kind) in enumerate(all_symbols):
        try:
            key = (resolve_equity_instrument_key(symbol, token) if kind == "equity"
                   else resolve_futures_instrument_key(symbol, token))
            if key is None:
                continue
            daily_df = fetch_candles(key, token, "days", "1", DAILY_LOOKBACK_DAYS)
            intraday_df = fetch_candles(key, token, "minutes", "5", COMPOSITE_LOOKBACK_DAYS)

            prev_close = float(daily_df["close"].iloc[-1]) if not daily_df.empty else None
            avg_daily_volume = (float(daily_df["volume"].tail(RVOL_BASELINE_DAYS).mean())
                                 if len(daily_df) >= RVOL_BASELINE_DAYS else None)
            composite_zones = compute_composite_zones(intraday_df)
            intraday_zones = compute_intraday_zones(intraday_df)  # seed with today's slice of what we already have

            cache[symbol] = {
                "instrument_key": key,
                "prev_close": prev_close,
                "avg_daily_volume": avg_daily_volume,
                "composite_zones": composite_zones,
                "intraday_zones": intraday_zones,
                "last_signal": "-",
                "zones_updated_at": now_ist().strftime("%Y-%m-%d %H:%M:%S"),
                "today_events": [],  # reset each day at Precompute -- see run_live_scan
            }
        except Exception as e:
            st.warning(f"{symbol}: precompute failed ({e}), skipping.")
        if progress_callback:
            progress_callback(i + 1, len(all_symbols), symbol)
        time.sleep(0.15)

    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)
    return cache


def run_zone_refresh(cache, token, progress_callback=None):
    """The 'medium' refresh tier: re-fetches TODAY's candles per symbol
    and recomputes intraday_zones. Composite zones are left untouched
    (those only change at the next Precompute)."""
    symbols = list(cache.keys())
    for i, symbol in enumerate(symbols):
        try:
            key = cache[symbol]["instrument_key"]
            today_df = fetch_today_candles(key, token)
            cache[symbol]["intraday_zones"] = compute_intraday_zones(today_df)
            cache[symbol]["zones_updated_at"] = now_ist().strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:
            st.warning(f"{symbol}: zone refresh failed ({e}), keeping previous zones.")
        if progress_callback:
            progress_callback(i + 1, len(symbols), symbol)
        time.sleep(0.1)

    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)
    return cache


def fetch_batch_quotes(instrument_keys, token):
    headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
    params = {"instrument_key": ",".join(instrument_keys)}
    resp = requests.get(QUOTES_URL, headers=headers, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json().get("data", {})


def nearest_zones(ltp, validated_zones):
    """Splits validated zones into support-side (price_mode <= ltp) and
    resistance-side (price_mode > ltp), and returns whichever of each is
    CLOSEST to ltp, along with the % distance from ltp to that zone's
    near edge (price_high for support, price_low for resistance -- the
    edge price would actually touch first)."""
    support, support_dist = None, None
    resistance, resistance_dist = None, None
    for z in validated_zones:
        if ltp is None:
            break
        if z["price_mode"] <= ltp:
            dist = abs(ltp - z["price_high"]) / ltp * 100
            if support_dist is None or dist < support_dist:
                support, support_dist = z, dist
        else:
            dist = abs(z["price_low"] - ltp) / ltp * 100
            if resistance_dist is None or dist < resistance_dist:
                resistance, resistance_dist = z, dist
    return support, support_dist, resistance, resistance_dist


def crossed_zones(prev_ltp, ltp, validated_zones):
    """Detects a zone LEVEL actually being crossed between the previous
    and current scan tick -- no VWAP condition, no 'near' threshold,
    fires the instant price crosses through a validated zone's price_mode.

    Direction determines the label, and it works out cleanly with no
    extra classification needed:
      - price FALLS through a level (prev_ltp >= level > ltp): that level
        is now above current price, i.e. it's acting as resistance going
        forward -- this is a "Resistance breakdown" (bearish).
      - price RISES through a level (prev_ltp < level <= ltp): that level
        is now below current price, i.e. it's acting as support going
        forward -- this is a "Support reclaim" (bullish).
    This matches the same after-the-fact support/resistance labeling the
    chart itself uses (zone vs. current price), so an alert fired here
    lines up with what you'd see if you opened the Chart tab right after.
    """
    breakdowns = []  # price fell through a level -> now resistance overhead
    reclaims = []     # price rose through a level -> now support underneath
    if prev_ltp is None or ltp is None:
        return breakdowns, reclaims
    for z in validated_zones:
        level = z["price_mode"]
        if prev_ltp >= level > ltp:
            breakdowns.append(z)
        elif prev_ltp < level <= ltp:
            reclaims.append(z)
    return breakdowns, reclaims


def build_setup_display_df(rows, zone_kind):
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if zone_kind == "support":
        df = df[["symbol", "ltp", "vwap", "zone_level", "zone_pct", "distance_pct",
                  "room_level", "room_pct"]]
        df.columns = ["Symbol", "LTP", "VWAP", "Support level", "Zone %", "Distance %",
                      "Next resistance", "Room to run %"]
        # best risk/reward (most room before hitting resistance) floats to top;
        # symbols with no resistance overhead at all show blank Room and sort last
        return df.sort_values("Room to run %", ascending=False, na_position="last").reset_index(drop=True)
    else:
        df = df[["symbol", "ltp", "vwap", "zone_level", "zone_pct", "distance_pct",
                  "room_level", "room_pct"]]
        df.columns = ["Symbol", "LTP", "VWAP", "Resistance level", "Zone %", "Distance %",
                      "Next support", "Room to fall %"]
        return df.sort_values("Room to fall %", ascending=False, na_position="last").reset_index(drop=True)


def build_level_cross_display_df(rows, kind):
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if kind == "breakdown":
        df = df[["symbol", "ltp", "level", "zone_pct", "next_level", "next_pct"]]
        df.columns = ["Symbol", "LTP", "Broke below", "Zone %", "Next support", "Room to fall %"]
        return df.sort_values("Room to fall %", ascending=False, na_position="last").reset_index(drop=True)
    else:
        df = df[["symbol", "ltp", "level", "zone_pct", "next_level", "next_pct"]]
        df.columns = ["Symbol", "LTP", "Broke above", "Zone %", "Next resistance", "Room to run %"]
        return df.sort_values("Room to run %", ascending=False, na_position="last").reset_index(drop=True)


MIN_ROOM_PCT_FOR_TOP_BOXES = 1.0  # only surface level-breaks with >1% room to move


def filter_by_room(rows, min_room_pct):
    """Keeps a level-break row if it has enough room to the next zone to be
    worth acting on, OR if there's no zone at all on that side (None = open
    air, which is arguably the best case, not a bad one -- so it passes
    through rather than getting filtered out for 'missing' data)."""
    return [r for r in rows if r.get("next_pct") is None or r["next_pct"] > min_room_pct]


def run_live_scan(cache, token):
    """Fast tier: one batch quote call for LTP/VWAP, signal recomputed
    against whichever zones are currently cached (may be a few minutes
    stale if 'Refresh Zones' hasn't been run recently). Also computes RVOL
    (today's volume so far / prior-N-day average full-day volume) and
    ranks the top TOP_N_RVOL symbols -- only those are eligible to alert
    (see update_alert_log), since unusual volume is the conviction filter
    that keeps alerts to a handful of genuinely active names instead of
    every symbol that happens to tick across VWAP."""
    symbols = list(cache.keys())
    instrument_keys = [cache[s]["instrument_key"] for s in symbols]
    key_to_symbol = {cache[s]["instrument_key"]: s for s in symbols}
    quotes = fetch_batch_quotes(instrument_keys, token)

    rows = []
    signals = {}
    bottom_setups = []       # near support + just crossed above VWAP
    top_setups = []          # near resistance + just closed below VWAP
    resistance_breakdowns = []  # pure level cross: price fell through a validated zone
    support_reclaims = []       # pure level cross: price rose through a validated zone

    for quote_key, q in quotes.items():
        instrument_key = q.get("instrument_token")
        symbol = key_to_symbol.get(instrument_key)
        if not symbol:
            continue
        c = cache[symbol]
        ltp = q.get("last_price")
        vwap = q.get("average_price")
        today_volume = q.get("volume")
        prev_close = c.get("prev_close")
        avg_daily_volume = c.get("avg_daily_volume")

        change_pct = (round((ltp - prev_close) / prev_close * 100, 2)
                      if ltp is not None and prev_close else None)
        rvol_pct = (round(today_volume / avg_daily_volume * 100, 1)
                    if today_volume is not None and avg_daily_volume else None)
        signal = compute_zone_signal(
            ltp, vwap, c.get("composite_zones", []), c.get("intraday_zones", []),
            min_distance_pct=MIN_SIGNAL_DISTANCE_PCT,
            min_vwap_distance_pct=MIN_VWAP_DISTANCE_PCT,
        )
        signals[symbol] = {"signal": signal, "ltp": ltp, "vwap": vwap, "rvol_pct": rvol_pct}
        cache[symbol]["last_signal"] = signal

        # --- Setup detection: near support/resistance + VWAP cross ---
        # "Just crossed" is edge-triggered off the PREVIOUS scan's
        # above/below state, persisted in the cache (same pattern as
        # update_alert_log's edge-triggering below) so it survives
        # across reruns instead of re-firing on every refresh.
        if ltp is not None:
            val_comp, _, _ = cross_validated_zones(
                c.get("composite_zones", []), c.get("intraday_zones", [])
            )
            support, support_dist, resistance, resistance_dist = nearest_zones(ltp, val_comp)

            if vwap is not None:
                vwap_above_now = ltp > vwap
                prev_vwap_above = c.get("prev_vwap_above")
                crossed_up = prev_vwap_above is False and vwap_above_now
                crossed_down = prev_vwap_above is True and not vwap_above_now
                cache[symbol]["prev_vwap_above"] = vwap_above_now

                if support is not None and support_dist is not None and support_dist <= NEAR_ZONE_PCT and crossed_up:
                    bottom_setups.append({
                        "symbol": symbol, "ltp": ltp, "vwap": round(vwap, 2),
                        "zone_level": support["price_mode"],
                        "zone_pct": _pct_from_label_safe(support["label"]),
                        "distance_pct": round(support_dist, 2),
                        # "room to run": distance from LTP up to the next resistance
                        # overhead. None means no validated resistance zone was found
                        # above current price at all -- i.e. open air, arguably the
                        # BEST case for a bounce, not a bad one, so it's not filtered
                        # out, just shown blank and sorted to the bottom by default.
                        "room_level": resistance["price_mode"] if resistance is not None else None,
                        "room_pct": round(resistance_dist, 2) if resistance_dist is not None else None,
                    })

                if resistance is not None and resistance_dist is not None and resistance_dist <= NEAR_ZONE_PCT and crossed_down:
                    top_setups.append({
                        "symbol": symbol, "ltp": ltp, "vwap": round(vwap, 2),
                        "zone_level": resistance["price_mode"],
                        "zone_pct": _pct_from_label_safe(resistance["label"]),
                        "distance_pct": round(resistance_dist, 2),
                        # "room to fall": distance from LTP down to the next support
                        # floor. None means no validated support zone found below --
                        # i.e. open air on the downside if this rejection plays out.
                        "room_level": support["price_mode"] if support is not None else None,
                        "room_pct": round(support_dist, 2) if support_dist is not None else None,
                    })

            # --- Pure level-cross detection: no VWAP condition, no "near"
            # threshold. Fires the instant LTP actually crosses a validated
            # zone's price_mode, which catches moves as they start (e.g. a
            # breakdown at the open) rather than waiting for a VWAP
            # confirmation that might come minutes or hours later. ---
            prev_ltp = c.get("prev_ltp")
            level_breakdowns, level_reclaims = crossed_zones(prev_ltp, ltp, val_comp)
            cache[symbol]["prev_ltp"] = ltp

            for z in level_breakdowns:
                # after breaking down through this level, the nearest
                # zone still below current price is the next support --
                # i.e. the next likely target/floor if the move continues.
                next_support, next_support_dist, _, _ = nearest_zones(ltp, val_comp)
                resistance_breakdowns.append({
                    "symbol": symbol, "ltp": ltp,
                    "level": z["price_mode"],
                    "zone_pct": _pct_from_label_safe(z["label"]),
                    "next_level": next_support["price_mode"] if next_support is not None else None,
                    "next_pct": round(next_support_dist, 2) if next_support_dist is not None else None,
                })
                # record with a timestamp so the Chart tab can mark the
                # exact candle this fired on, tying the Setups-tab table
                # to a specific point on the chart instead of leaving it
                # only as a row in a separate table.
                cache[symbol].setdefault("today_events", []).append({
                    "time": now_ist().strftime("%Y-%m-%d %H:%M:%S"),
                    "label": f"Broke below {z['price_mode']:.0f}",
                    "color": "#EF5350",
                })

            for z in level_reclaims:
                # after breaking up through this level, the nearest zone
                # still above current price is the next resistance -- the
                # next hurdle if the move continues.
                _, _, next_resistance, next_resistance_dist = nearest_zones(ltp, val_comp)
                support_reclaims.append({
                    "symbol": symbol, "ltp": ltp,
                    "level": z["price_mode"],
                    "zone_pct": _pct_from_label_safe(z["label"]),
                    "next_level": next_resistance["price_mode"] if next_resistance is not None else None,
                    "next_pct": round(next_resistance_dist, 2) if next_resistance_dist is not None else None,
                })
                cache[symbol].setdefault("today_events", []).append({
                    "time": now_ist().strftime("%Y-%m-%d %H:%M:%S"),
                    "label": f"Broke above {z['price_mode']:.0f}",
                    "color": "#26A69A",
                })

        rows.append({
            "Symbol": symbol, "PrevClose": prev_close, "LTP": ltp,
            "Change%": change_pct, "VWAP": vwap, "RVOL%": rvol_pct, "Signal": signal,
        })

    ranked = sorted(
        [(s, d["rvol_pct"]) for s, d in signals.items() if d["rvol_pct"] is not None],
        key=lambda x: x[1], reverse=True,
    )
    top_n_symbols = set(s for s, _ in ranked[:TOP_N_RVOL])

    df = pd.DataFrame(rows)
    if not df.empty:
        df["Top5RVOL"] = df["Symbol"].isin(top_n_symbols)
        df = df.sort_values("RVOL%", ascending=False, na_position="last").reset_index(drop=True)
        df.insert(0, "S.No", range(1, len(df) + 1))
    return df, signals, top_n_symbols, bottom_setups, top_setups, resistance_breakdowns, support_reclaims


def _pct_from_label_safe(label):
    m = re.search(r"[\d.]+", str(label))
    return float(m.group()) if m else 0.0


def load_alert_log():
    if os.path.exists(ALERT_LOG_PATH):
        with open(ALERT_LOG_PATH, "r") as f:
            return json.load(f)
    return {"alerts": [], "last_eligible_symbols": []}


def save_alert_log(log):
    with open(ALERT_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)


def update_alert_log(alert_log, signals, eligible_symbols):
    """Edge-triggered: only logs a new entry the moment a symbol's signal
    changes to a fresh BUY/SELL state, not on every refresh it stays
    active. Gated to market hours -- an after-hours refresh pulls Upstox's
    frozen post-close LTP/VWAP, which must not get logged as a live
    signal (same bug already fixed in hvn-lvn-scanner's paper trader).

    Also gated to eligible_symbols (the current top TOP_N_RVOL by RVOL).
    "Newly entered" (just entered the top N this cycle) is tracked via
    alert_log["last_eligible_symbols"], persisted to disk -- NOT via
    st.session_state. Session state is per-browser-session, so a page
    reload or a Streamlit Cloud reconnect resets it to empty, making
    every currently-eligible symbol look "newly entered" again and
    re-logging duplicates seconds after the original (this exact bug
    was seen live: 5 symbols logged twice, 4 seconds apart). Persisting
    to the same file the dedup check already reads from survives
    reconnects correctly."""
    now = now_ist()
    prev_eligible = set(alert_log.get("last_eligible_symbols", []))
    newly_entered = eligible_symbols - prev_eligible
    alert_log["last_eligible_symbols"] = list(eligible_symbols)

    market_is_open = MARKET_OPEN_TIME <= now.time() < MARKET_CLOSE_TIME
    if not market_is_open:
        return alert_log

    last_signal = {a["symbol"]: a["signal"] for a in alert_log["alerts"] if a.get("is_latest")}
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    for symbol, data in signals.items():
        if symbol not in eligible_symbols:
            continue
        sig = data["signal"]
        if sig not in ("BUY", "SELL"):
            continue
        is_fresh_entry = symbol in newly_entered
        if not is_fresh_entry and last_signal.get(symbol) == sig:
            continue
        for a in alert_log["alerts"]:
            if a["symbol"] == symbol:
                a["is_latest"] = False
        alert_log["alerts"].append({
            "symbol": symbol, "signal": sig, "ltp": data["ltp"], "vwap": data["vwap"],
            "rvol_pct": data.get("rvol_pct"), "time": now_str, "is_latest": True,
        })
    return alert_log


def build_alert_display_df(alert_log, price_lookup=None):
    """price_lookup: {symbol: current LTP}, from the latest Scanner scan --
    used to mark-to-market each alert's PnL% against its entry price
    (the LTP captured at the moment the alert fired)."""
    if not alert_log["alerts"]:
        return pd.DataFrame()
    price_lookup = price_lookup or {}
    df = pd.DataFrame(alert_log["alerts"])
    if "rvol_pct" not in df.columns:
        df["rvol_pct"] = None

    def _current_price(row):
        return price_lookup.get(row["symbol"])

    def _pnl(row):
        cur = row["current_ltp"]
        entry = row["ltp"]
        if cur is None or entry is None:
            return None
        if row["signal"] == "BUY":
            return round((cur - entry) / entry * 100, 2)
        elif row["signal"] == "SELL":
            return round((entry - cur) / entry * 100, 2)
        return None

    df["current_ltp"] = df.apply(_current_price, axis=1)
    df["pnl_pct"] = df.apply(_pnl, axis=1)
    df = df[["symbol", "signal", "ltp", "current_ltp", "pnl_pct", "vwap", "rvol_pct", "time"]]
    df.columns = ["Symbol", "Signal", "EntryPrice", "LTP", "PnL%", "VWAP", "RVOL%", "Time"]
    df = df.sort_values("Time", ascending=False).reset_index(drop=True)
    df.insert(0, "S.No", range(1, len(df) + 1))
    return df


def build_zones_display_df(zones):
    if not zones:
        return pd.DataFrame()
    df = pd.DataFrame(zones)
    df = df[["price_mode", "label", "price_low", "price_high"]]
    df.columns = ["Level", "Zone %", "Range Low", "Range High"]
    return df.sort_values("Level", ascending=False).reset_index(drop=True)


def shared_y_range(dfs, zone_lists):
    """Union of price extents across multiple panels' candle data AND
    their zones, so a horizontal support/resistance line lines up at the
    same height across panels sharing this range -- e.g. the 18-day
    composite panel and today's developing panel for the same stock --
    for direct visual comparison instead of two independently-scaled
    charts."""
    lows, highs = [], []
    for df in dfs:
        if df is not None and not df.empty:
            lows.append(df["low"].min())
            highs.append(df["high"].max())
    for zones in zone_lists:
        for z in zones or []:
            lows.append(z["price_low"])
            highs.append(z["price_high"])
    if not lows or not highs:
        return None
    lo, hi = min(lows), max(highs)
    pad = (hi - lo) * 0.05 if hi > lo else 1
    return (lo - pad, hi + pad)


def build_screener_df(cache, price_lookup):
    """Ranks EVERY symbol by distance to its single nearest validated
    zone (support or resistance, whichever is closer), ascending -- the
    stocks genuinely at a decision point right now float to the top,
    instead of having to scroll sector by sector hoping to spot one."""
    rows = []
    for symbol, c in cache.items():
        ltp = price_lookup.get(symbol)
        if ltp is None:
            continue
        val_comp, _, _ = cross_validated_zones(
            c.get("composite_zones", []), c.get("intraday_zones", [])
        )
        support, support_dist, resistance, resistance_dist = nearest_zones(ltp, val_comp)

        # pick whichever side is actually closer
        if support_dist is not None and (resistance_dist is None or support_dist <= resistance_dist):
            nearest, dist, kind = support, support_dist, "Support"
        elif resistance_dist is not None:
            nearest, dist, kind = resistance, resistance_dist, "Resistance"
        else:
            continue  # no validated zone on either side at all

        rows.append({
            "Symbol": symbol, "Sector": SECTOR_MAP.get(symbol, "-"),
            "LTP": ltp, "Nearest level": nearest["price_mode"],
            "Kind": kind, "Distance %": round(dist, 2),
            "Zone %": _pct_from_label_safe(nearest["label"]),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("Distance %").reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_composite_history_cached(instrument_key, token, days):
    """Cached for an hour -- this is the SAME 5-min data
    compute_composite_zones() used to build the composite zones in
    run_precompute, so re-fetching it constantly on every grid re-render
    (every auto-refresh tick, every tab switch) would be wasteful for
    data that only actually changes once a day at the next Precompute."""
    return fetch_candles(instrument_key, token, "minutes", "5", days)


def render_symbol_grid(cache, symbols_list, token, key_prefix="sector"):
    # One stock per row, two panels: 18-day composite (left, at 5-min
    # resolution -- the same data the zones were built from) and today's
    # developing session (right) -- stack stocks vertically so scrolling
    # moves through the whole sector. Used by both the Sectors tab and
    # the Support/Resistance Screener tabs (jump-to-sector view).
    for sym in symbols_list:
        c = cache[sym]
        st.markdown(f"**{sym}**")
        col_left, col_right = st.columns(2)

        val_comp, _, _ = cross_validated_zones(
            c.get("composite_zones", []), c.get("intraday_zones", [])
        )

        composite_df = fetch_composite_history_cached(
            c["instrument_key"], token, COMPOSITE_LOOKBACK_DAYS
        )
        grid_df = get_today_candles(sym, c["instrument_key"], token)

        with col_left:
            if composite_df.empty:
                st.write("No history available.")
            else:
                fig_left = plot_candles_with_zones(
                    composite_df,
                    composite_zones=c.get("composite_zones", []),
                    intraday_zones=[],
                    validated_zones=val_comp,
                    title=f"{sym} - 18 days (5-min)",
                    height=220, compact=True,
                    show_vwap=False, tick_format="%d %b",
                    market_hours_breaks=True,
                )
                st.plotly_chart(fig_left, use_container_width=True,
                                 key=f"{key_prefix}_composite_{sym}")

        with col_right:
            if grid_df.empty:
                st.write("No candle data yet for today.")
            else:
                fig_right = plot_candles_with_zones(
                    grid_df,
                    composite_zones=[],  # avoid clutter in the compact grid cell
                    intraday_zones=[],
                    validated_zones=val_comp,
                    title=f"{sym} - today",
                    height=220, compact=True,
                )
                st.plotly_chart(fig_right, use_container_width=True,
                                 key=f"{key_prefix}_today_{sym}")

        st.divider()


# ---------------- UI (four tabs: Scanner, Key Levels, Chart, Alerts) ----------------
st.set_page_config(page_title="Sahi Key Levels LIVE", layout="wide")
st.title("Sahi Key Levels LIVE")
st.caption(
    "Cross-timeframe validated zones: a level only counts if BOTH the 18-day composite "
    "profile and today's intraday profile independently show volume clustered there."
)

col1, col2, col3 = st.columns(3)
run_precompute_clicked = col1.button("Run Precompute (slow, once/day)")
refresh_zones_clicked = col2.button("Refresh Zones (medium, every few min)")
refresh_quotes_clicked = col3.button("Refresh Quotes (fast)")

auto_refresh_enabled = st.checkbox(
    "Auto-refresh (quotes every 1 min, zones every 5 min) - only while this tab stays open",
    value=False,
)
auto_tick = st_autorefresh(interval=AUTO_REFRESH_QUOTES_SECONDS * 1000, key="auto_refresh_tick") if auto_refresh_enabled else None
if "last_auto_tick" not in st.session_state:
    st.session_state["last_auto_tick"] = -1
auto_quotes_due = auto_tick is not None and auto_tick != st.session_state["last_auto_tick"]
if auto_quotes_due:
    st.session_state["last_auto_tick"] = auto_tick
auto_zone_due = auto_quotes_due and auto_tick > 0 and auto_tick % ZONE_REFRESH_EVERY_N_TICKS == 0

if run_precompute_clicked:
    token = get_token()
    progress_bar = st.progress(0)
    status_text = st.empty()
    def _cb(i, total, symbol):
        progress_bar.progress(i / total)
        status_text.text(f"{i}/{total}: {symbol}")
    with st.spinner("Running precompute..."):
        cache = run_precompute(token, progress_callback=_cb)
    st.success(f"Precompute done. {len(cache)} symbols cached.")

if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "r") as f:
        cache = json.load(f)

    if refresh_zones_clicked or auto_zone_due:
        token = get_token()
        progress_bar = st.progress(0)
        status_text = st.empty()
        def _cb2(i, total, symbol):
            progress_bar.progress(i / total)
            status_text.text(f"{i}/{total}: {symbol}")
        with st.spinner("Refreshing intraday zones..."):
            cache = run_zone_refresh(cache, token, progress_callback=_cb2)
        st.success("Zone refresh done.")

    if refresh_quotes_clicked or refresh_zones_clicked or auto_quotes_due or auto_zone_due or "last_scan_df" not in st.session_state:
        token = get_token()
        scan_df, signals, top_n_symbols, bottom_setups, top_setups, resistance_breakdowns, support_reclaims = run_live_scan(cache, token)

        alert_log = load_alert_log()
        alert_log = update_alert_log(alert_log, signals, top_n_symbols)
        save_alert_log(alert_log)

        with open(CACHE_PATH, "w") as f:
            json.dump(cache, f)

        st.session_state["last_scan_df"] = scan_df
        st.session_state["last_refresh_time"] = now_ist().strftime("%H:%M:%S")
        st.session_state["alert_log"] = alert_log
        st.session_state["bottom_setups"] = bottom_setups
        st.session_state["top_setups"] = top_setups
        st.session_state["resistance_breakdowns"] = resistance_breakdowns
        st.session_state["support_reclaims"] = support_reclaims

    df = st.session_state.get("last_scan_df", pd.DataFrame())
    price_lookup = dict(zip(df["Symbol"], df["LTP"])) if not df.empty else {}
    alert_log = st.session_state.get("alert_log") or load_alert_log()

    symbols_with_zones = [s for s in cache if cache[s].get("composite_zones") or cache[s].get("intraday_zones")]
    default_idx = symbols_with_zones.index("NIFTY") if "NIFTY" in symbols_with_zones else 0

    # --- Always-visible top summary: level breaks with real room to move ---
    # Sits above the tabs so it's visible no matter which tab is open --
    # the whole point is to avoid scrolling through 220 sector charts to
    # find what's actionable right now.
    top_breakdowns = filter_by_room(st.session_state.get("resistance_breakdowns", []), MIN_ROOM_PCT_FOR_TOP_BOXES)
    top_reclaims = filter_by_room(st.session_state.get("support_reclaims", []), MIN_ROOM_PCT_FOR_TOP_BOXES)

    box_col1, box_col2 = st.columns(2)
    with box_col1:
        st.markdown(f"**🔴 Resistance breakdown** (>{MIN_ROOM_PCT_FOR_TOP_BOXES:.0f}% room to fall)")
        bd_df = build_level_cross_display_df(top_breakdowns, "breakdown")
        if bd_df.empty:
            st.caption("None this cycle.")
        else:
            st.dataframe(bd_df, use_container_width=True, hide_index=True, height=150)
    with box_col2:
        st.markdown(f"**🟢 Support reclaim** (>{MIN_ROOM_PCT_FOR_TOP_BOXES:.0f}% room to run)")
        rc_df = build_level_cross_display_df(top_reclaims, "reclaim")
        if rc_df.empty:
            st.caption("None this cycle.")
        else:
            st.dataframe(rc_df, use_container_width=True, hide_index=True, height=150)

    st.divider()

    tab_scanner, tab_support_screener, tab_resistance_screener, tab_levels, tab_chart, tab_sectors, tab_setups, tab_replay, tab_alerts = st.tabs(
        ["Scanner", "Support Screener", "Resistance Screener", "Key Levels", "Chart", "Sectors", "Setups", "Replay", "Alerts"]
    )

    def render_screener_tab(kind_label):
        """kind_label: 'Support' or 'Resistance' -- filters build_screener_df's
        ranked list to just that side, then lets you pick a stock from it to
        see its ENTIRE sector rendered right here (same dual-chart view as
        the Sectors tab). Streamlit has no way to programmatically jump to a
        different tab, so this is the equivalent: pick a stock, see its
        sector peers immediately, without leaving this tab."""
        st.caption(
            f"Symbols ranked by how close they are to a validated {kind_label.lower()} "
            f"level -- closest (most actionable) first."
        )
        full_df = build_screener_df(cache, price_lookup)
        screener_df = full_df[full_df["Kind"] == kind_label].reset_index(drop=True) if not full_df.empty else full_df

        if screener_df.empty:
            st.write(f"No validated {kind_label.lower()} zones with a live price yet - "
                     f"click 'Refresh Quotes'.")
            return

        n_show = st.slider(
            "Show top N", min_value=10, max_value=len(screener_df),
            value=min(30, len(screener_df)), key=f"screener_n_{kind_label}",
        )
        shown_df = screener_df.head(n_show)
        st.dataframe(shown_df, use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("**See this stock's sector**")
        symbol_to_sector = dict(zip(shown_df["Symbol"], shown_df["Sector"]))
        pick_symbol = st.selectbox(
            "Pick a stock from the table above to see its whole sector "
            "(all peer stocks, same dual-chart view as the Sectors tab)",
            shown_df["Symbol"].tolist(), key=f"screener_pick_{kind_label}",
            format_func=lambda sym: f"{sym} — {symbol_to_sector.get(sym, '-')}",
        )
        if pick_symbol:
            pick_sector = SECTOR_MAP.get(pick_symbol)
            if pick_sector is None:
                st.write(f"{pick_symbol} isn't mapped to a sector.")
            else:
                sector_peers = [s for s in symbols_with_zones if SECTOR_MAP.get(s) == pick_sector]
                st.markdown(f"### {pick_sector} (via {pick_symbol})")
                token = get_token()
                render_symbol_grid(cache, sector_peers, token, key_prefix=f"screener_{kind_label}")

    with tab_support_screener:
        render_screener_tab("Support")

    with tab_resistance_screener:
        render_screener_tab("Resistance")

    with tab_scanner:
        st.caption(f"Last refreshed: {st.session_state.get('last_refresh_time', 'never')}")
        if df.empty:
            st.write("No data yet - click Refresh Quotes.")
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)

    with tab_levels:
        st.caption(
            "Composite = 18-day profile (updates on Precompute). Intraday = today's session "
            "(updates on Refresh Zones). Only overlapping ranges across both count as validated."
        )
        if not symbols_with_zones:
            st.write("No zones available yet - click 'Run Precompute'.")
        else:
            selected_symbol = st.selectbox("Symbol", symbols_with_zones, index=default_idx, key="levels_symbol")
            c = cache[selected_symbol]
            st.caption(f"Zones last updated: {c.get('zones_updated_at', 'never')}")

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Composite (18-day)**")
                st.dataframe(build_zones_display_df(c.get("composite_zones", [])),
                             use_container_width=True, hide_index=True)
            with col_b:
                st.markdown("**Intraday (today)**")
                st.dataframe(build_zones_display_df(c.get("intraday_zones", [])),
                             use_container_width=True, hide_index=True)

            val_comp, val_intra, _ = cross_validated_zones(
                c.get("composite_zones", []), c.get("intraday_zones", [])
            )
            st.markdown("**Validated (confirmed by both timeframes)**")
            validated_display = build_zones_display_df(val_comp)
            if validated_display.empty:
                st.write("No cross-validated zones yet.")
            else:
                st.dataframe(validated_display, use_container_width=True, hide_index=True)

    with tab_chart:
        if not symbols_with_zones:
            st.write("No zones available yet - click 'Run Precompute'.")
        else:
            chart_symbol = st.selectbox("Symbol", symbols_with_zones, index=default_idx, key="chart_symbol")
            c = cache[chart_symbol]
            token = get_token()

            val_comp, val_intra, _ = cross_validated_zones(
                c.get("composite_zones", []), c.get("intraday_zones", [])
            )

            # 5-min candles over the composite lookback window -- the SAME
            # data compute_composite_zones() was built from in run_precompute,
            # so the candles shown here match exactly what produced these
            # zones, at real intraday resolution instead of one blunt daily
            # bar per day.
            composite_hist_df = fetch_candles(
                c["instrument_key"], token, "minutes", "5", COMPOSITE_LOOKBACK_DAYS
            )
            chart_df = get_today_candles(chart_symbol, c["instrument_key"], token)

            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown(f"**Previous {COMPOSITE_LOOKBACK_DAYS} days (composite, 5-min)**")
                if composite_hist_df.empty:
                    st.write("No history available.")
                else:
                    fig_left = plot_candles_with_zones(
                        composite_hist_df,
                        composite_zones=c.get("composite_zones", []),
                        intraday_zones=[],
                        validated_zones=val_comp,
                        title=f"{chart_symbol} - last {COMPOSITE_LOOKBACK_DAYS} days (5-min)",
                        show_vwap=False,  # a single cumulative VWAP across 18 days isn't meaningful
                        tick_format="%d %b",
                        market_hours_breaks=True,
                    )
                    st.plotly_chart(fig_left, use_container_width=True, key="chart_composite_left")

            with col_right:
                st.markdown("**Today (developing)**")
                if chart_df.empty:
                    st.write("No candle data yet for today.")
                else:
                    # Turn today's recorded level-break events into chart
                    # markers -- only ones that fall within the currently
                    # plotted session (today_events accumulates across the
                    # whole day; a stale event from before market open on
                    # a prior run shouldn't show if it's somehow outside
                    # the visible candle range).
                    session_start = chart_df["timestamp"].iloc[0]
                    session_end = chart_df["timestamp"].iloc[-1]
                    chart_events = []
                    for ev in c.get("today_events", []):
                        try:
                            ev_time = pd.Timestamp(ev["time"])
                            # match tz-awareness to whatever chart_df's
                            # timestamps actually are, rather than assuming --
                            # Upstox's raw timestamp format can come back
                            # either way depending on parse path.
                            if session_start.tzinfo is not None and ev_time.tzinfo is None:
                                ev_time = ev_time.tz_localize(IST)
                            elif session_start.tzinfo is None and ev_time.tzinfo is not None:
                                ev_time = ev_time.tz_localize(None)
                        except Exception:
                            continue
                        if session_start <= ev_time <= session_end:
                            chart_events.append({
                                "time": ev_time, "label": ev["label"], "color": ev.get("color"),
                            })

                    fig_right = plot_candles_with_zones(
                        chart_df,
                        composite_zones=c.get("composite_zones", []),
                        intraday_zones=c.get("intraday_zones", []),
                        validated_zones=val_comp,
                        title=f"{chart_symbol} - price with key levels",
                        event_markers=chart_events,
                    )
                    st.plotly_chart(fig_right, use_container_width=True, key="chart_today_right")

    with tab_sectors:
        st.caption(
            "Scan by sector to see who's sitting at support, who's stuck at resistance, "
            "and who's in open air. Each stock shows two panels: the last 18 days on the "
            "left, today's developing session on the right -- scroll down to move through "
            "every stock in the sector."
        )
        available_sectors = sorted(set(
            SECTOR_MAP[s] for s in symbols_with_zones if s in SECTOR_MAP
        ))
        if not available_sectors:
            st.write("No sector data available yet - click 'Run Precompute'.")
        else:
            view_mode = st.radio(
                "View", ["One sector at a time", "Scroll through all sectors"],
                horizontal=True, key="sector_view_mode",
            )

            if view_mode == "One sector at a time":
                selected_sector = st.selectbox("Sector", available_sectors, key="sector_select")
                sector_symbols = [s for s in symbols_with_zones if SECTOR_MAP.get(s) == selected_sector]

                if not sector_symbols:
                    st.write("No symbols with zones in this sector yet.")
                else:
                    token = get_token()
                    render_symbol_grid(cache, sector_symbols, token)

            else:
                total_symbols = len([s for s in symbols_with_zones if s in SECTOR_MAP])
                st.caption(
                    f"Renders all {total_symbols} symbols across {len(available_sectors)} sectors "
                    f"in one continuous scroll -- this fetches a lot of candle data at once, so it's "
                    f"gated behind this button rather than running automatically. Once loaded, "
                    f"results are cached for 45s so switching tabs or auto-refresh ticks won't "
                    f"immediately re-fetch everything."
                )
                if st.button("Load all sectors", key="load_all_sectors"):
                    st.session_state["show_all_sectors"] = True

                if st.session_state.get("show_all_sectors"):
                    token = get_token()
                    for sector in available_sectors:
                        sector_symbols = [s for s in symbols_with_zones if SECTOR_MAP.get(s) == sector]
                        if not sector_symbols:
                            continue
                        st.markdown(f"## {sector}")
                        render_symbol_grid(cache, sector_symbols, token)
                        st.divider()

    with tab_setups:
        st.markdown("### Level breaks (fires the instant price crosses a level -- no VWAP needed)")
        st.caption(
            "Catches a move as it happens, e.g. a breakdown right at the open, instead of "
            "waiting for a VWAP confirmation that might come much later in the session."
        )
        st.markdown("**Resistance breakdown** (price just fell through a validated level -- bearish)")
        breakdown_df = build_level_cross_display_df(st.session_state.get("resistance_breakdowns", []), "breakdown")
        if breakdown_df.empty:
            st.write("None this cycle.")
        else:
            st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

        st.markdown("**Support reclaim** (price just rose through a validated level -- bullish)")
        reclaim_df = build_level_cross_display_df(st.session_state.get("support_reclaims", []), "reclaim")
        if reclaim_df.empty:
            st.write("None this cycle.")
        else:
            st.dataframe(reclaim_df, use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("### VWAP-confirmed setups")
        st.caption(
            f"Edge-triggered: a symbol appears only the cycle it happens, not every refresh "
            f"it stays true. 'Near' means within {NEAR_ZONE_PCT}% of the validated zone's edge. "
            f"'Room to run/fall' is the distance to the NEXT zone on the opposite side -- more "
            f"room means more space for the move to actually play out before hitting resistance "
            f"(or support, for the rejection table). Blank room = no validated zone found that "
            f"direction at all, i.e. open air."
        )
        st.markdown("**At support, just crossed above VWAP** (possible bounce)")
        bottom_df = build_setup_display_df(st.session_state.get("bottom_setups", []), "support")
        if bottom_df.empty:
            st.write("None this cycle.")
        else:
            st.dataframe(bottom_df, use_container_width=True, hide_index=True)

        st.markdown("**At resistance, just closed below VWAP** (possible rejection)")
        top_df = build_setup_display_df(st.session_state.get("top_setups", []), "resistance")
        if top_df.empty:
            st.write("None this cycle.")
        else:
            st.dataframe(top_df, use_container_width=True, hide_index=True)

    with tab_replay:
        st.caption(
            "Pick a past trading day and scrub (or press Play) to watch candles, intraday "
            "zones, and validated zones form exactly as they would have appeared live -- "
            "same logic as backtest_zone_formation.py, but visual instead of console output."
        )
        if not symbols_with_zones:
            st.write("No zones available yet - click 'Run Precompute'.")
        else:
            replay_mode = st.radio(
                "Mode", ["Single symbol", "Whole sector together"],
                horizontal=True, key="replay_mode",
            )
            replay_date = st.date_input(
                "Trading day", value=now_ist().date() - timedelta(days=1),
                max_value=now_ist().date() - timedelta(days=1), key="replay_date",
            )
            token = get_token()

            if replay_mode == "Single symbol":
                replay_symbol = st.selectbox("Symbol", symbols_with_zones, index=default_idx, key="replay_symbol")
                c = cache[replay_symbol]

                composite_zones, day_df = fetch_replay_data(
                    c["instrument_key"], token, replay_date.strftime("%Y-%m-%d")
                )

                if day_df is None or day_df.empty:
                    st.write("No candle data for that day (market holiday, weekend, or not enough "
                             "prior history for a composite window). Try a different date.")
                else:
                    n_candles = len(day_df)

                    # Reset playback when symbol/date changes; clamp a
                    # stale slider value from a previous day's different
                    # candle count.
                    selection_id = f"single_{replay_symbol}_{replay_date}"
                    if st.session_state.get("replay_selection_id") != selection_id:
                        st.session_state["replay_selection_id"] = selection_id
                        st.session_state["replay_slider"] = 1
                        st.session_state["replay_playing"] = False
                    elif st.session_state.get("replay_slider", 1) > n_candles:
                        st.session_state["replay_slider"] = n_candles

                    play_col, restart_col, speed_col = st.columns([1, 1, 2])
                    with play_col:
                        playing = st.checkbox("Play", key="replay_playing")
                    with restart_col:
                        if st.button("Restart", key="replay_restart_single"):
                            st.session_state["replay_slider"] = 1
                    with speed_col:
                        speed_ms = st.select_slider(
                            "Speed", options=[1000, 600, 300, 150], value=600,
                            format_func=lambda v: f"{1000 / v:.1f}x", key="replay_speed",
                        )

                    if playing:
                        st_autorefresh(interval=speed_ms, key="replay_autoplay_tick")
                        if st.session_state["replay_slider"] < n_candles:
                            st.session_state["replay_slider"] += 1
                        else:
                            st.session_state["replay_playing"] = False

                    slider_pos = st.slider(
                        "Candles shown (scrub through the session)",
                        min_value=1, max_value=n_candles, key="replay_slider",
                    )
                    so_far = day_df.iloc[:slider_pos].reset_index(drop=True)
                    intraday_zones = compute_intraday_zones(so_far)
                    val_comp, _, _ = cross_validated_zones(composite_zones, intraday_zones)

                    current_time = so_far["timestamp"].iloc[-1].strftime("%H:%M")
                    st.caption(f"Showing up to {current_time} -- {len(val_comp)} validated zone(s) "
                               f"at this point in the session.")

                    # Pin the x-axis to the FULL session's span so candles
                    # stay properly sized from the very first slider
                    # position, instead of a lone early candle stretching
                    # to fill the whole chart width.
                    full_session_range = (day_df["timestamp"].iloc[0], day_df["timestamp"].iloc[-1])

                    fig = plot_candles_with_zones(
                        so_far,
                        composite_zones=composite_zones,
                        intraday_zones=intraday_zones,
                        validated_zones=val_comp,
                        title=f"{replay_symbol} replay - {replay_date}",
                        x_range=full_session_range,
                    )
                    st.plotly_chart(fig, use_container_width=True, key="replay_chart_single")

            else:  # Whole sector together
                available_sectors = sorted(set(
                    SECTOR_MAP[s] for s in symbols_with_zones if s in SECTOR_MAP
                ))
                if not available_sectors:
                    st.write("No sector data available yet - click 'Run Precompute'.")
                else:
                    selected_sector = st.selectbox("Sector", available_sectors, key="replay_sector_select")
                    sector_symbols = [s for s in symbols_with_zones if SECTOR_MAP.get(s) == selected_sector]

                    if not sector_symbols:
                        st.write("No symbols with zones in this sector yet.")
                    else:
                        # Fetch each symbol's replay data -- fetch_replay_data
                        # is cached per (symbol, date), so re-picking the
                        # same sector/date later doesn't re-fetch anything.
                        per_symbol_data = {}
                        max_candles = 0
                        for sym in sector_symbols:
                            c = cache[sym]
                            comp_zones, day_df_sym = fetch_replay_data(
                                c["instrument_key"], token, replay_date.strftime("%Y-%m-%d")
                            )
                            if day_df_sym is not None and not day_df_sym.empty:
                                per_symbol_data[sym] = (comp_zones, day_df_sym)
                                max_candles = max(max_candles, len(day_df_sym))

                        if not per_symbol_data:
                            st.write("No candle data for any symbol in this sector on that day. "
                                     "Try a different date.")
                        else:
                            selection_id = f"sector_{selected_sector}_{replay_date}"
                            if st.session_state.get("replay_selection_id") != selection_id:
                                st.session_state["replay_selection_id"] = selection_id
                                st.session_state["replay_slider_sector"] = 1
                                st.session_state["replay_playing_sector"] = False
                            elif st.session_state.get("replay_slider_sector", 1) > max_candles:
                                st.session_state["replay_slider_sector"] = max_candles

                            play_col, restart_col, speed_col = st.columns([1, 1, 2])
                            with play_col:
                                playing_s = st.checkbox("Play", key="replay_playing_sector")
                            with restart_col:
                                if st.button("Restart", key="replay_restart_sector"):
                                    st.session_state["replay_slider_sector"] = 1
                            with speed_col:
                                speed_ms_s = st.select_slider(
                                    "Speed", options=[1000, 600, 300, 150], value=600,
                                    format_func=lambda v: f"{1000 / v:.1f}x", key="replay_speed_sector",
                                )

                            if playing_s:
                                st_autorefresh(interval=speed_ms_s, key="replay_autoplay_tick_sector")
                                if st.session_state["replay_slider_sector"] < max_candles:
                                    st.session_state["replay_slider_sector"] += 1
                                else:
                                    st.session_state["replay_playing_sector"] = False

                            slider_pos_s = st.slider(
                                "Candles shown (all symbols in this sector advance together)",
                                min_value=1, max_value=max_candles, key="replay_slider_sector",
                            )

                            st.markdown(f"## {selected_sector} -- {replay_date}")
                            symbols_list = list(per_symbol_data.keys())
                            for i in range(0, len(symbols_list), 2):
                                row_symbols = symbols_list[i:i + 2]
                                cols = st.columns(len(row_symbols))
                                for col, sym in zip(cols, row_symbols):
                                    with col:
                                        comp_zones, day_df_sym = per_symbol_data[sym]
                                        # a symbol with fewer candles than
                                        # max_candles (e.g. a late listing
                                        # or a data gap) just stays at its
                                        # own last available candle rather
                                        # than erroring
                                        pos = min(slider_pos_s, len(day_df_sym))
                                        so_far_sym = day_df_sym.iloc[:pos].reset_index(drop=True)
                                        intraday_zones_sym = compute_intraday_zones(so_far_sym)
                                        val_comp_sym, _, _ = cross_validated_zones(comp_zones, intraday_zones_sym)
                                        full_range_sym = (day_df_sym["timestamp"].iloc[0],
                                                           day_df_sym["timestamp"].iloc[-1])
                                        fig_sym = plot_candles_with_zones(
                                            so_far_sym,
                                            composite_zones=[], intraday_zones=[],
                                            validated_zones=val_comp_sym,
                                            title=sym, height=260, compact=True,
                                            x_range=full_range_sym,
                                        )
                                        st.plotly_chart(fig_sym, use_container_width=True,
                                                         key=f"replay_sector_chart_{sym}")

    with tab_alerts:
        st.caption(
            f"Logged the moment a symbol's signal changes to a fresh BUY/SELL - not repeated "
            f"every refresh it stays active. Only the top {TOP_N_RVOL} symbols by RVOL are eligible "
            f"to alert. Only logged during market hours (9:15-15:30 IST)."
        )
        alert_df = build_alert_display_df(alert_log, price_lookup)
        if alert_df.empty:
            st.write("No alerts logged yet.")
        else:
            st.dataframe(alert_df, use_container_width=True, hide_index=True)
            csv = alert_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download alert log CSV", csv, "alert_log.csv", "text/csv")
else:
    st.info("No cache found yet - click 'Run Precompute' first.")
