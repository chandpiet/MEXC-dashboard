#!/usr/bin/env python3
"""
MEXC Stock Futures S/R Dashboard v3 — Maximum Quality + Hybrid Pricing
======================================================================
Ultra-selective S/R + entry scanner for MEXC USDT-M Stock Futures.

Core features:
- Strict multi-TF confluence (Daily + 4H + HVN)
- Trend / regime filter (SMA50 + SPY bias)
- Volume spike + candle rejection confirmation
- Tight distance, strength, quality score, and R:R filters
- Position sizing calculator
- CSV export + Discord webhook alerts
- Market-hours / data-freshness flag
- Hybrid pricing:
    * Yahoo Finance underlying → zones, ATR, swings, volume profile
    * Live MEXC futures price → displayed Last + distance-to-zone
- Fair Value Gap (FVG) detection on Daily + 4H (confluence + standalone)

Run:
  streamlit run mexc_sr_dashboard.py --server.port 8501
"""

from __future__ import annotations

import json
import math
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, time as dtime
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks
import streamlit as st

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------

UNIVERSE: Dict[str, str] = {
    "NVDAUSDT": "NVDA", "AMDUSDT": "AMD", "MUUSDT": "MU", "AVGOUSDT": "AVGO",
    "MRVLUSDT": "MRVL", "ASMLUSDT": "ASML", "ARMUSDT": "ARM", "QCOMUSDT": "QCOM",
    "INTCUSDT": "INTC", "SMCIUSDT": "SMCI", "SNDKUSDT": "SNDK", "LITEUSDT": "LITE",
    "AAOIUSDT": "AAOI", "TSLAUSDT": "TSLA", "AAPLUSDT": "AAPL", "MSFTUSDT": "MSFT",
    "METAUSDT": "META", "AMZNUSDT": "AMZN", "GOOGLUSDT": "GOOGL", "PLTRUSDT": "PLTR",
    "NFLXUSDT": "NFLX", "ORCLUSDT": "ORCL", "ADBEUSDT": "ADBE", "CSCOUSDT": "CSCO",
    "UBERUSDT": "UBER", "APPUSDT": "APP", "COINUSDT": "COIN", "HOODUSDT": "HOOD",
    "MSTRUSDT": "MSTR", "CRCLUSDT": "CRCL", "LLYUSDT": "LLY", "GEUSDT": "GE",
    "MCDUSDT": "MCD", "PEPUSDT": "PEP", "UNHUSDT": "UNH", "VUSDT": "V", "MAUSDT": "MA",
    "SP500USDT": "SPY", "SPYUSDT": "SPY", "NAS100USDT": "QQQ", "QQQUSDT": "QQQ",
    "US30USDT": "DIA", "SOXLUSDT": "SOXL",
}

# MEXC contract symbol mapping (for live futures price)
MEXC_SYMBOL: Dict[str, str] = {
    "NVDAUSDT": "NVIDIA_USDT", "TSLAUSDT": "TESLA_USDT", "METAUSDT": "METASTOCK_USDT",
    "AAPLUSDT": "AAPLSTOCK_USDT", "AMDUSDT": "AMDSTOCK_USDT", "MSFTUSDT": "MSFTSTOCK_USDT",
    "AMZNUSDT": "AMZNSTOCK_USDT", "GOOGLUSDT": "GOOGLSTOCK_USDT", "PLTRUSDT": "PLTRSTOCK_USDT",
    "HOODUSDT": "ROBINHOOD_USDT", "ARMUSDT": "ARMSTOCK_USDT", "INTCUSDT": "INTCSTOCK_USDT",
    "MUUSDT": "MUSTOCK_USDT", "AVGOUSDT": "AVGOSTOCK_USDT", "SMCIUSDT": "SMCISTOCK_USDT",
    "QCOMUSDT": "QCOMSTOCK_USDT", "MRVLUSDT": "MRVLSTOCK_USDT", "ASMLUSDT": "ASMLSTOCK_USDT",
    "NFLXUSDT": "NFLXSTOCK_USDT", "ORCLUSDT": "ORCLSTOCK_USDT", "ADBEUSDT": "ADBESTOCK_USDT",
    "CSCOUSDT": "CSCOSTOCK_USDT", "UBERUSDT": "UBERSTOCK_USDT", "APPUSDT": "APPSTOCK_USDT",
    "MSTRUSDT": "MSTRSTOCK_USDT", "CRCLUSDT": "CRCLSTOCK_USDT", "LLYUSDT": "LLYSTOCK_USDT",
    "GEUSDT": "GESTOCK_USDT", "MCDUSDT": "MCDSTOCK_USDT", "PEPUSDT": "PEPSTOCK_USDT",
    "UNHUSDT": "UNHSTOCK_USDT", "VUSDT": "VSTOCK_USDT", "MAUSDT": "MASTOCK_USDT",
    "SNDKUSDT": "SNDKSTOCK_USDT", "LITEUSDT": "LITESTOCK_USDT", "AAOIUSDT": "AAOISTOCK_USDT",
    "SP500USDT": "SPY_USDT", "SPYUSDT": "SPY_USDT", "NAS100USDT": "NAS100_USDT",
    "QQQUSDT": "QQQSTOCK_USDT",
}

# Tight parameters
ATR_PERIOD = 14
PEAK_DIST_D = 5
PEAK_DIST_4H = 7
PROM_MULT = 0.85
ZONE_EPS_PCT = 0.004
ZONE_EPS_ATR = 0.28
MIN_TOUCHES = 2
RECENCY_HALFLIFE = 35
VOL_WEIGHT = 0.45
FIB_LEVELS = [0.382, 0.5, 0.618]
FIB_TOL = 0.006
ATR_STOP_MULT = 0.50
MIN_RR = 2.2
VP_BINS = 40
HVN_PCT = 78
MAX_DIST_ATR = 0.55          # very tight proximity
MIN_STRENGTH = 3.2
MIN_QUALITY = 7.0            # composite score out of ~12
VOL_SPIKE = 1.40
SMA_LEN = 50

# Discord webhook is pulled from Streamlit secrets (set in App Settings → Secrets),
# never hardcoded here, since this repo is public.
DISCORD_WEBHOOK = st.secrets.get("DISCORD_WEBHOOK", "")


def is_nyse_regular_hours() -> Tuple[bool, str]:
    """Return (is_open, status_message). NYSE regular ~ 9:30–16:00 America/New_York."""
    try:
        now = datetime.now(ZoneInfo("America/New_York"))
        t = now.time()
        weekday = now.weekday()  # 0=Mon ... 6=Sun
        if weekday >= 5:
            return False, "Weekend — underlying equity data is stale (last close)"
        open_t = dtime(9, 30)
        close_t = dtime(16, 0)
        if open_t <= t <= close_t:
            return True, "NYSE regular hours — data should be updating"
        if t < open_t:
            return False, "Pre-market — underlying data may be stale or limited"
        return False, "After-hours / evening — underlying equity data is stale (last close)"
    except Exception:
        return True, "Unable to determine market hours"


def calc_position_size(account: float, risk_pct: float, entry: float, stop: float) -> dict:
    """Simple position size for USDT-margined futures (approximate contracts / notional)."""
    if account <= 0 or risk_pct <= 0 or entry <= 0 or stop <= 0 or entry == stop:
        return {"risk_amount": 0.0, "size_coins": 0.0, "notional": 0.0}
    risk_amount = account * (risk_pct / 100.0)
    risk_per_unit = abs(entry - stop)
    size = risk_amount / risk_per_unit if risk_per_unit > 0 else 0.0
    notional = size * entry
    return {
        "risk_amount": round(risk_amount, 2),
        "size_coins": round(size, 4),
        "notional": round(notional, 2),
    }


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Zone:
    low: float
    high: float
    center: float
    touches: int
    total_volume: float
    strength: float
    kind: str
    source: str = "swing"
    last_touch_idx: int = 0
    fib: bool = False
    tf: str = "1D"


@dataclass
class EntryIdea:
    side: str
    entry: float
    stop: float
    target: float
    rr: float
    reason: str
    zone_center: float
    style: str
    quality: float
    dist_atr: float


@dataclass
class AnalysisResult:
    futures: str
    equity: str
    last: float                  # Yahoo underlying (used for zones)
    mexc_last: Optional[float] = None  # Live MEXC futures price
    atr_d: float = 0.0
    atr_4h: float = 0.0
    sma50: float = 0.0
    sma50_rising: bool = False
    zones: List[Zone] = field(default_factory=list)
    entries: List[EntryIdea] = field(default_factory=list)
    df_d: Optional[pd.DataFrame] = None
    nearest_zone_dist_atr: float = 99.0
    market_bias: str = "neutral"
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Fetching (robust)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60, show_spinner=False)
def fetch_mexc_price(futures_sym: str) -> Optional[float]:
    """Live last price from MEXC stock futures contract."""
    mexc_sym = MEXC_SYMBOL.get(futures_sym)
    if not mexc_sym:
        return None
    try:
        url = f"https://contract.mexc.com/api/v1/contract/ticker?symbol={mexc_sym}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as r:
            data = json.loads(r.read())
        if data.get("success") and data.get("data"):
            return float(data["data"]["lastPrice"])
    except Exception:
        pass
    return None


@st.cache_data(ttl=180, show_spinner=False)
def fetch_ohlcv(ticker: str, range_: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    for attempt in range(3):
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval={interval}&range={range_}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; research)"})
            with urllib.request.urlopen(req, timeout=15) as r:
                data = json.loads(r.read())
            res = data["chart"]["result"][0]
            ts = res["timestamp"]
            q = res["indicators"]["quote"][0]
            df = pd.DataFrame({
                "Open": q["open"], "High": q["high"], "Low": q["low"],
                "Close": q["close"], "Volume": q["volume"]
            }, index=pd.to_datetime(ts, unit="s"))
            df = df.dropna()
            df = df[df["Volume"] > 0]
            return df
        except Exception:
            time.sleep(0.8 * (attempt + 1))
    return pd.DataFrame()


def to_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    if df_1h.empty:
        return df_1h
    return df_1h.resample("4h").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
    }).dropna()


def atr(series_df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    h, l, c = series_df["High"], series_df["Low"], series_df["Close"]
    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


# ---------------------------------------------------------------------------
# Volume Profile HVN
# ---------------------------------------------------------------------------

def volume_profile_hvn(df: pd.DataFrame, bins: int = VP_BINS, pct: float = HVN_PCT) -> List[Tuple[float, float, float]]:
    if len(df) < 25:
        return []
    pmin, pmax = df["Low"].min(), df["High"].max()
    if pmax <= pmin:
        return []
    edges = np.linspace(pmin, pmax, bins + 1)
    volb = np.zeros(bins)
    for _, row in df.iterrows():
        lo, hi, v = row["Low"], row["High"], row["Volume"]
        if hi <= lo:
            idx = min(bins - 1, max(0, np.searchsorted(edges, lo, side="right") - 1))
            volb[idx] += v
            continue
        for i in range(bins):
            o = max(0.0, min(hi, edges[i + 1]) - max(lo, edges[i]))
            if o > 0:
                volb[i] += v * (o / (hi - lo))
    thr = np.percentile(volb[volb > 0], pct) if np.any(volb > 0) else 0
    hvns = []
    i = 0
    while i < bins:
        if volb[i] >= thr and volb[i] > 0:
            j = i
            while j + 1 < bins and volb[j + 1] >= thr:
                j += 1
            hvns.append((float(edges[i]), float(edges[j + 1]), float(volb[i:j + 1].sum())))
            i = j + 1
        else:
            i += 1
    return hvns


def detect_fair_value_gaps(df: pd.DataFrame, atr_val: float, tf: str = "1D",
                          min_gap_atr: float = 0.25) -> List[Zone]:
    """
    Detect 3-candle Fair Value Gaps (ICT-style imbalances).
    """
    if len(df) < 10:
        return []
    gaps: List[Zone] = []
    highs = df["High"].values
    lows = df["Low"].values
    n = len(df)

    for i in range(2, n):
        # Bullish FVG (gap up)
        if highs[i - 2] < lows[i]:
            gap_low = float(highs[i - 2])
            gap_high = float(lows[i])
            gap_size = gap_high - gap_low
            if gap_size < atr_val * min_gap_atr:
                continue
            filled = False
            for j in range(i + 1, n):
                if lows[j] <= gap_low:
                    filled = True
                    break
            if filled:
                continue
            center = (gap_low + gap_high) / 2
            gaps.append(Zone(
                low=gap_low, high=gap_high, center=center,
                touches=2, total_volume=0.0, strength=0.0,
                kind="support", source="fvg", last_touch_idx=i, tf=tf
            ))

        # Bearish FVG (gap down)
        if lows[i - 2] > highs[i]:
            gap_high = float(lows[i - 2])
            gap_low = float(highs[i])
            gap_size = gap_high - gap_low
            if gap_size < atr_val * min_gap_atr:
                continue
            filled = False
            for j in range(i + 1, n):
                if highs[j] >= gap_high:
                    filled = True
                    break
            if filled:
                continue
            center = (gap_low + gap_high) / 2
            gaps.append(Zone(
                low=gap_low, high=gap_high, center=center,
                touches=2, total_volume=0.0, strength=0.0,
                kind="resistance", source="fvg", last_touch_idx=i, tf=tf
            ))

    return gaps


# ---------------------------------------------------------------------------
# Swings & Zones
# ---------------------------------------------------------------------------

def find_swings(df: pd.DataFrame, atr_s: pd.Series, distance: int):
    highs, lows = df["High"].values, df["Low"].values
    med = float(np.nanmedian(atr_s.values[-70:])) if len(atr_s) > 20 else float(np.nanmedian(atr_s))
    if not med or np.isnan(med) or med <= 0:
        med = 1e-6
    prom = max(med * PROM_MULT, 1e-8)
    hi, _ = find_peaks(highs, distance=distance, prominence=prom)
    lo, _ = find_peaks(-lows, distance=distance, prominence=prom)
    return hi, highs[hi], lo, lows[lo]


def cluster(prices, vols, idxs, eps, kind, tf) -> List[Zone]:
    if len(prices) == 0:
        return []
    order = np.argsort(prices)
    prices, vols, idxs = prices[order], vols[order], idxs[order]
    clusters = [[0]]
    for i in range(1, len(prices)):
        if prices[i] - prices[clusters[-1][-1]] <= eps:
            clusters[-1].append(i)
        else:
            clusters.append([i])
    out = []
    for cl in clusters:
        p, v, ix = prices[cl], vols[cl], idxs[cl]
        center = float(np.average(p, weights=v + 1e-9))
        low, high = float(p.min()), float(p.max())
        if high - low < 1e-9:
            half = eps * 0.4
            low -= half
            high += half
        out.append(Zone(low=low, high=high, center=center, touches=len(cl),
                        total_volume=float(v.sum()), strength=0.0, kind=kind,
                        source="swing", last_touch_idx=int(ix.max()), tf=tf))
    return out


def score_zones(zones: List[Zone], df: pd.DataFrame, sig: Optional[Tuple[float, float]]) -> List[Zone]:
    n = len(df)
    vmed = float(df["Volume"].median()) + 1e-9
    scored = []
    for z in zones:
        age = n - 1 - z.last_touch_idx
        rec = math.exp(-math.log(2) * age / RECENCY_HALFLIFE)
        volf = 1.0 + VOL_WEIGHT * math.log1p(z.total_volume / vmed)
        base = z.touches * volf * rec
        fibb = 0.0
        if sig:
            lo, hi = sig
            rng = hi - lo
            if rng > 0:
                for f in FIB_LEVELS:
                    for lvl in (hi - f * rng, lo + f * rng):
                        if abs(z.center - lvl) / max(z.center, 1e-9) < FIB_TOL:
                            fibb = 2.0
                            z.fib = True
                            break
        z.strength = base + fibb
        if z.source == "hvn":
            z.strength *= 1.35
        if z.source == "fvg":
            z.strength = max(z.strength, 3.0) + 1.2
        if z.source == "both" and "fvg" in str(getattr(z, "source", "")):
            z.strength *= 1.25
        scored.append(z)
    scored = [z for z in scored if z.touches >= MIN_TOUCHES or z.strength > 2.5
              or z.source in ("hvn", "fvg")]
    scored.sort(key=lambda x: -x.strength)
    return scored


def merge_close(zones: List[Zone], tol: float = 0.0035) -> List[Zone]:
    if not zones:
        return []
    zones = sorted(zones, key=lambda z: z.center)
    merged = [zones[0]]
    for z in zones[1:]:
        last = merged[-1]
        if abs(z.center - last.center) / max(last.center, 1e-9) < tol:
            nl, nh = min(last.low, z.low), max(last.high, z.high)
            nv = last.total_volume + z.total_volume
            nc = (last.center * last.total_volume + z.center * z.total_volume) / (nv + 1e-9)
            kind = "both" if last.kind != z.kind else last.kind
            src = "both" if last.source != z.source else last.source
            tf = "MTF" if last.tf != z.tf else last.tf
            merged[-1] = Zone(low=nl, high=nh, center=nc, touches=last.touches + z.touches,
                              total_volume=nv, strength=max(last.strength, z.strength),
                              kind=kind, source=src, last_touch_idx=max(last.last_touch_idx, z.last_touch_idx),
                              fib=last.fib or z.fib, tf=tf)
        else:
            merged.append(z)
    return merged


# ---------------------------------------------------------------------------
# Quality scoring & ultra-tight entries
# ---------------------------------------------------------------------------

def candle_rejection(df: pd.DataFrame, side: str) -> bool:
    """Simple rejection / close strength on last bar."""
    if len(df) < 3:
        return False
    o, h, l, c = df["Open"].iloc[-1], df["High"].iloc[-1], df["Low"].iloc[-1], df["Close"].iloc[-1]
    body = abs(c - o)
    rng = h - l + 1e-9
    if side == "long":
        lower_wick = min(o, c) - l
        return (lower_wick > body * 1.1) or (c > o and (c - l) / rng > 0.65)
    else:
        upper_wick = h - max(o, c)
        return (upper_wick > body * 1.1) or (c < o and (h - c) / rng > 0.65)


def generate_entries(zones: List[Zone], last: float, atr: float, df: pd.DataFrame,
                     sma50: float, sma_rising: bool, market_bias: str) -> List[EntryIdea]:
    ideas = []
    if not atr or atr <= 0:
        return []
    atr_buf = atr * ATR_STOP_MULT
    strong = [z for z in zones if z.strength >= MIN_STRENGTH][:8]
    if not strong:
        return []

    avg_vol = df["Volume"].iloc[-20:].mean() + 1e-9
    last_vol = df["Volume"].iloc[-1]
    vol_ok = last_vol >= avg_vol * VOL_SPIKE

    for z in strong:
        dist = abs(last - z.center) / atr
        if dist > MAX_DIST_ATR:
            continue

        # --- LONG bounce ---
        if z.kind in ("support", "both") and last >= z.low - atr * 0.15:
            if last > z.high + atr * 0.8:
                continue
            if not (last > sma50 and sma_rising):
                continue
            if market_bias == "risk_off":
                continue
            if not candle_rejection(df, "long") and not vol_ok:
                continue

            entry = max(z.high, last * 0.999) + atr * 0.08
            stop = z.low - atr_buf
            risk = entry - stop
            if risk <= 0:
                continue
            higher = [zz for zz in strong if zz.center > entry + atr * 0.4]
            target = higher[0].low if higher else entry + risk * MIN_RR
            rr = (target - entry) / risk
            if rr < MIN_RR * 0.95:
                continue

            q = 0.0
            q += min(z.strength / 2.0, 3.5)
            q += 2.0 if z.source in ("hvn", "both", "fvg") else 0.0
            q += 1.5 if z.tf in ("MTF", "4H") else 0.5
            q += 1.5 if z.fib else 0.0
            q += 1.5 if vol_ok else 0.0
            q += 1.0 if candle_rejection(df, "long") else 0.0
            q += 1.0 if dist < 0.35 else 0.0
            q += 1.0 if market_bias == "risk_on" else 0.0
            if q < MIN_QUALITY:
                continue

            ideas.append(EntryIdea("long", round(entry, 4), round(stop, 4), round(target, 4),
                                   round(rr, 2), f"Bounce S={z.strength:.1f} {z.source}/{z.tf}",
                                   z.center, "bounce", round(q, 1), round(dist, 2)))

        # --- SHORT bounce ---
        if z.kind in ("resistance", "both") and last <= z.high + atr * 0.15:
            if last < z.low - atr * 0.8:
                continue
            if not (last < sma50 and not sma_rising):
                continue
            if market_bias == "risk_on":
                continue
            if not candle_rejection(df, "short") and not vol_ok:
                continue

            entry = min(z.low, last * 1.001) - atr * 0.08
            stop = z.high + atr_buf
            risk = stop - entry
            if risk <= 0:
                continue
            lower = [zz for zz in strong if zz.center < entry - atr * 0.4]
            target = lower[0].high if lower else entry - risk * MIN_RR
            rr = (entry - target) / risk
            if rr < MIN_RR * 0.95:
                continue

            q = 0.0
            q += min(z.strength / 2.0, 3.5)
            q += 2.0 if z.source in ("hvn", "both", "fvg") else 0.0
            q += 1.5 if z.tf in ("MTF", "4H") else 0.5
            q += 1.5 if z.fib else 0.0
            q += 1.5 if vol_ok else 0.0
            q += 1.0 if candle_rejection(df, "short") else 0.0
            q += 1.0 if dist < 0.35 else 0.0
            q += 1.0 if market_bias == "risk_off" else 0.0
            if q < MIN_QUALITY:
                continue

            ideas.append(EntryIdea("short", round(entry, 4), round(stop, 4), round(target, 4),
                                   round(rr, 2), f"Bounce R={z.strength:.1f} {z.source}/{z.tf}",
                                   z.center, "bounce", round(q, 1), round(dist, 2)))

    ideas.sort(key=lambda e: (-e.quality, -e.rr))
    return ideas[:2]


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def get_market_bias() -> str:
    try:
        spy = fetch_ohlcv("SPY", "3mo", "1d")
        if len(spy) < 60:
            return "neutral"
        sma = spy["Close"].rolling(SMA_LEN).mean()
        last = spy["Close"].iloc[-1]
        rising = sma.iloc[-1] > sma.iloc[-5]
        if last > sma.iloc[-1] and rising:
            return "risk_on"
        if last < sma.iloc[-1] and not rising:
            return "risk_off"
        return "neutral"
    except Exception:
        return "neutral"


def analyze(futures: str, equity: str, market_bias: str, fvg_enabled: bool = True) -> AnalysisResult:
    try:
        df_d = fetch_ohlcv(equity, "6mo", "1d")
        if len(df_d) < 70:
            return AnalysisResult(futures=futures, equity=equity, last=0, error="Insufficient data — try refreshing")

        df_1h = fetch_ohlcv(equity, "60d", "1h")
        df_4h = to_4h(df_1h) if len(df_1h) > 40 else None

        atr_d_s = atr(df_d)
        atr_d = float(atr_d_s.iloc[-1]) if len(atr_d_s) else 0.0
        if not atr_d or atr_d <= 0 or pd.isna(atr_d):
            return AnalysisResult(futures=futures, equity=equity, last=0,
                                   error="Data unavailable (flat/stale feed) — try refreshing")
        last = float(df_d["Close"].iloc[-1])
        if not last or last <= 0 or pd.isna(last):
            return AnalysisResult(futures=futures, equity=equity, last=0,
                                   error="Data unavailable (bad price) — try refreshing")

        sma50_s = df_d["Close"].rolling(SMA_LEN).mean()
        sma50 = float(sma50_s.iloc[-1]) if not pd.isna(sma50_s.iloc[-1]) else last
        sma_rising = bool(sma50_s.iloc[-1] > sma50_s.iloc[-6]) if len(sma50_s) > 6 and not pd.isna(sma50_s.iloc[-6]) else False

        # Daily swings + HVN
        h_idx, h_p, l_idx, l_p = find_swings(df_d, atr_d_s, PEAK_DIST_D)
        vol = df_d["Volume"].values
        eps = max(last * ZONE_EPS_PCT, atr_d * ZONE_EPS_ATR, 1e-6)
        zones = cluster(h_p, vol[h_idx], h_idx, eps, "resistance", "1D") + \
                cluster(l_p, vol[l_idx], l_idx, eps, "support", "1D")
        for lo, hi, v in volume_profile_hvn(df_d.iloc[-80:]):
            zones.append(Zone(low=lo, high=hi, center=(lo + hi) / 2, touches=3,
                              total_volume=v, strength=0.0, kind="both", source="hvn",
                              last_touch_idx=len(df_d) - 3, tf="1D"))

        if fvg_enabled:
            fvg_d = detect_fair_value_gaps(df_d.iloc[-120:], atr_d, tf="1D", min_gap_atr=0.28)
            zones.extend(fvg_d)

        recent = df_d.iloc[-90:]
        sig = (float(recent["Low"].min()), float(recent["High"].max()))
        zones = score_zones(zones, df_d, sig)

        # 4H
        atr_4h = atr_d
        if df_4h is not None and len(df_4h) > 50:
            atr4_s = atr(df_4h)
            atr_4h_val = float(atr4_s.iloc[-1]) if len(atr4_s) else 0.0
            if atr_4h_val and not pd.isna(atr_4h_val) and atr_4h_val > 0:
                atr_4h = atr_4h_val
            h4, hp4, l4, lp4 = find_swings(df_4h, atr4_s, PEAK_DIST_4H)
            v4 = df_4h["Volume"].values
            eps4 = max(last * ZONE_EPS_PCT * 0.9, atr_4h * ZONE_EPS_ATR, 1e-6)
            z4 = cluster(hp4, v4[h4], h4, eps4, "resistance", "4H") + \
                 cluster(lp4, v4[l4], l4, eps4, "support", "4H")
            if fvg_enabled:
                fvg_4h = detect_fair_value_gaps(df_4h.iloc[-100:], atr_4h, tf="4H", min_gap_atr=0.30)
                z4.extend(fvg_4h)
            z4 = score_zones(z4, df_4h, sig)
            zones = merge_close(zones + z4)
        else:
            zones = merge_close(zones)

        # Extra confluence: boost any zone that overlaps a significant FVG
        fvg_zones = [z for z in zones if z.source == "fvg"]
        for z in zones:
            if z.source == "fvg":
                continue
            for f in fvg_zones:
                if z.low <= f.high and z.high >= f.low:
                    z.strength *= 1.30
                    z.source = "both" if z.source != "both" else z.source
                    break

        zones = sorted(zones, key=lambda z: -z.strength)[:14]

        # Live MEXC futures price (for display + distance)
        mexc_last = fetch_mexc_price(futures)
        price_for_dist = mexc_last if mexc_last is not None else last

        # Nearest zone distance (prefer MEXC price when available)
        nearest = 99.0
        for z in zones:
            d = abs(price_for_dist - z.center) / atr_d
            if d < nearest:
                nearest = d

        entries = generate_entries(zones, price_for_dist, atr_d, df_d, sma50, sma_rising, market_bias)

        return AnalysisResult(
            futures=futures, equity=equity, last=last, mexc_last=mexc_last,
            atr_d=atr_d, atr_4h=atr_4h, sma50=sma50, sma50_rising=sma_rising,
            zones=zones, entries=entries, df_d=df_d,
            nearest_zone_dist_atr=nearest, market_bias=market_bias
        )
    except ZeroDivisionError:
        return AnalysisResult(futures=futures, equity=equity, last=0,
                               error="Data unavailable (calc issue) — try refreshing")
    except Exception as e:
        return AnalysisResult(futures=futures, equity=equity, last=0,
                               error=f"Data unavailable — try refreshing ({type(e).__name__})")


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def make_chart(res: AnalysisResult) -> go.Figure:
    df = res.df_d
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28],
                        vertical_spacing=0.04)

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price", increasing_line_color="#26a69a", decreasing_line_color="#ef5350"
    ), row=1, col=1)

    sma20 = df["Close"].rolling(20).mean()
    sma50 = df["Close"].rolling(50).mean()
    fig.add_trace(go.Scatter(x=df.index, y=sma20, name="SMA20", line=dict(width=1.2, color="#42a5f5")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=sma50, name="SMA50", line=dict(width=1.5, color="#ffa726")), row=1, col=1)

    colors = {"support": "rgba(38,166,154,0.22)", "resistance": "rgba(239,83,80,0.22)", "both": "rgba(156,39,176,0.20)"}
    lc = {"support": "#26a69a", "resistance": "#ef5350", "both": "#ab47bc"}
    for z in res.zones[:8]:
        fig.add_hrect(y0=z.low, y1=z.high, fillcolor=colors.get(z.kind, "gray"), line_width=0, row=1, col=1)
        fig.add_hline(y=z.center, line_dash="dot", line_color=lc.get(z.kind, "gray"),
                      line_width=1.1, annotation_text=f"{z.kind[0].upper()}{z.strength:.1f}",
                      annotation_position="right", row=1, col=1)

    for e in res.entries:
        col = "#26a69a" if e.side == "long" else "#ef5350"
        fig.add_trace(go.Scatter(
            x=[df.index[-1]], y=[e.entry], mode="markers",
            marker=dict(symbol="triangle-up" if e.side == "long" else "triangle-down",
                        size=15, color=col, line=dict(width=1.5, color="white")),
            name=f"{e.side.upper()} Q={e.quality}",
            hovertext=f"{e.reason}<br>Q={e.quality} R:R={e.rr}<br>Dist {e.dist_atr} ATR"
        ), row=1, col=1)
        fig.add_hline(y=e.stop, line_dash="dash", line_color=col, line_width=1.2, row=1, col=1)
        fig.add_hline(y=e.target, line_dash="dashdot", line_color=col, line_width=1.2, row=1, col=1)

    colors_v = ["#26a69a" if c >= o else "#ef5350" for o, c in zip(df["Open"], df["Close"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=colors_v, name="Vol", opacity=0.45), row=2, col=1)

    fig.update_layout(
        title=(f"{res.futures} ({res.equity})  Yahoo {res.last:.2f}"
               + (f"  |  MEXC {res.mexc_last:.2f}" if res.mexc_last else "")
               + f"  |  ATR {res.atr_d:.2f}  |  Bias {res.market_bias}  |  Near {res.nearest_zone_dist_atr:.2f} ATR"),
        xaxis_rangeslider_visible=False, height=700, template="plotly_dark",
        margin=dict(l=50, r=40, t=55, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=10)),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="MEXC S/R v2 — Max Quality", page_icon="🎯", layout="wide",
                   initial_sidebar_state="expanded")

st.title("MEXC Stock Futures S/R — Maximum Quality + Trading Tools")
st.caption("Ultra-selective filters · Position sizing · CSV export · Market-hours flag · Optional webhook")

with st.sidebar:
    st.header("Controls")
    all_syms = sorted(UNIVERSE.keys())
    default = ["NVDAUSDT", "TSLAUSDT", "METAUSDT", "MUUSDT", "AAPLUSDT", "AMDUSDT",
               "PLTRUSDT", "HOODUSDT", "SP500USDT", "NAS100USDT", "ARMUSDT", "LLYUSDT"]
    selected = st.multiselect("Symbols", all_syms, default=[s for s in default if s in UNIVERSE])
    run_btn = st.button("🔄 Run Ultra-Selective Scan", type="primary", use_container_width=True)
    auto = st.checkbox("Auto-refresh every 4 min", value=False)
    fvg_enabled = st.checkbox("Enable Fair Value Gap detection", value=True,
                               help="Turn off to test whether FVG is related to any data errors on specific symbols.")

    st.markdown("---")
    st.subheader("Position Sizing")
    account_size = st.number_input("Account size (USDT)", min_value=100.0, value=5000.0, step=100.0)
    risk_pct = st.number_input("Risk per trade (%)", min_value=0.1, max_value=5.0, value=0.5, step=0.1)

    st.markdown("---")
    st.subheader("Alert Webhook")
    webhook_url = st.text_input(
        "Discord webhook URL",
        value=DISCORD_WEBHOOK,
        placeholder="https://discord.com/api/webhooks/..."
    )
    st.caption("Loaded from Streamlit secrets if set. You can also paste one here for this session only.")

    if webhook_url:
        if st.button("Send Test Message"):
            try:
                test_payload = {"content": "✅ Test message from your MEXC S/R Dashboard — webhook is working!"}
                req = urllib.request.Request(
                    webhook_url,
                    data=json.dumps(test_payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=8) as r:
                    if r.status in (200, 204):
                        st.success("Sent! Check your Discord channel.")
                    else:
                        st.error(f"Discord responded with status {r.status}")
            except Exception as e:
                st.error(f"Failed to send: {e}")
    else:
        st.caption("Paste or set a webhook URL above to enable test sending.")

    st.markdown("---")
    st.markdown("**Active Filters (all on)**")
    st.markdown("""
- Multi-TF / HVN / **FVG** confluence  
- SMA50 trend + rising  
- Market regime (SPY bias)  
- Volume spike ≥ 1.4×  
- Candle rejection  
- Dist ≤ 0.55 ATR  
- Strength ≥ 3.2  
- Quality score ≥ 7.0  
- Min R:R 2.2  
- Top 2 setups only  
""")
    st.markdown("---")
    st.markdown("Keep this tab open. Data from Yahoo Finance (underlying equities).")

if "results_v2" not in st.session_state:
    st.session_state.results_v2 = {}
    st.session_state.last_run_v2 = None
    st.session_state.bias = "neutral"

if run_btn or (auto and (st.session_state.last_run_v2 is None or
                         (datetime.now() - st.session_state.last_run_v2).seconds > 240)):
    if not selected:
        st.warning("Select symbols first.")
    else:
        bias = get_market_bias()
        st.session_state.bias = bias
        prog = st.progress(0.0, text="Running maximum-quality filters…")
        results = {}
        for i, sym in enumerate(selected):
            prog.progress((i + 0.4) / len(selected), text=f"Analyzing {sym}…")
            results[sym] = analyze(sym, UNIVERSE[sym], bias, fvg_enabled)
            time.sleep(0.4)
        st.session_state.results_v2 = results
        st.session_state.last_run_v2 = datetime.now()
        prog.progress(1.0, text="Done")
        time.sleep(0.25)
        prog.empty()

results = st.session_state.results_v2
bias = st.session_state.bias

if not results:
    st.info("Select symbols and click **Run Ultra-Selective Scan**.")
    st.stop()

is_open, hours_msg = is_nyse_regular_hours()
if is_open:
    st.success(f"**Data status:** {hours_msg}")
else:
    st.warning(f"**Data status:** {hours_msg}  \nMEXC futures trade 24/7, but underlying equity prices used for zones only update fully during NYSE regular hours. Treat after-hours levels with extra caution.")

bias_color = {"risk_on": "green", "risk_off": "red", "neutral": "orange"}
st.markdown(f"**Market Regime (SPY):** :{bias_color.get(bias, 'gray')}[{bias.upper()}]  —  Only setups aligned with this bias are shown.")

st.subheader("Ultra-Selective Scanner (Quality ≥ 7.0)")
rows = []
for sym, res in results.items():
    display_last = f"{res.mexc_last:.2f}" if res.mexc_last is not None else f"{res.last:.2f}"
    if res.mexc_last is not None:
        display_last = f"{res.mexc_last:.2f} (MEXC)"
    else:
        display_last = f"{res.last:.2f} (Y)"

    if res.error:
        rows.append({"Symbol": sym, "Last": "—", "Q": "", "Side": "ERR", "Entry": res.error[:45],
                     "Stop": "", "Target": "", "R:R": "", "Dist": "", "Size (coins)": "", "Notional": "", "Reason": ""})
        continue
    if not res.entries:
        near = f"{res.nearest_zone_dist_atr:.2f}" if res.nearest_zone_dist_atr < 10 else "—"
        rows.append({"Symbol": sym, "Last": display_last, "Q": "", "Side": "—", "Entry": "No high-Q setup",
                     "Stop": "", "Target": "", "R:R": "", "Dist": near, "Size (coins)": "", "Notional": "",
                     "Reason": f"Nearest zone {near} ATR"})
        continue
    for e in res.entries:
        ps = calc_position_size(account_size, risk_pct, e.entry, e.stop)
        rows.append({
            "Symbol": sym,
            "Last": display_last,
            "Q": e.quality,
            "Side": e.side.upper(),
            "Entry": f"{e.entry:.3f}",
            "Stop": f"{e.stop:.3f}",
            "Target": f"{e.target:.3f}",
            "R:R": e.rr,
            "Dist": e.dist_atr,
            "Size (coins)": ps["size_coins"],
            "Notional": ps["notional"],
            "Reason": e.reason,
        })

df_scan = pd.DataFrame(rows)
st.dataframe(df_scan, use_container_width=True, hide_index=True,
             column_config={
                 "Q": st.column_config.NumberColumn(format="%.1f"),
                 "R:R": st.column_config.NumberColumn(format="%.1f"),
                 "Dist": st.column_config.NumberColumn(format="%.2f"),
                 "Size (coins)": st.column_config.NumberColumn(format="%.4f"),
                 "Notional": st.column_config.NumberColumn(format="%.0f"),
             })

if not df_scan.empty:
    csv_buf = io.StringIO()
    df_scan.to_csv(csv_buf, index=False)
    st.download_button(
        label="📥 Download Scanner CSV (for journaling)",
        data=csv_buf.getvalue(),
        file_name=f"mexc_sr_scanner_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        use_container_width=False,
    )

if webhook_url and any(r.get("Q") for r in rows if isinstance(r.get("Q"), (int, float)) and r.get("Q", 0) >= 7):
    if st.button("Send high-Q setups to webhook"):
        try:
            payload = {"content": f"MEXC S/R high-Q setups:\n" + "\n".join(
                f"{r['Symbol']} {r['Side']} @ {r['Entry']} (Q={r['Q']}, R:R={r['R:R']})"
                for r in rows if isinstance(r.get("Q"), (int, float)) and r.get("Q", 0) >= 7
            )}
            req = urllib.request.Request(
                webhook_url, data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json", "User-Agent": "MEXC-SR-Dashboard"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                st.success(f"Webhook sent (status {resp.status})")
        except Exception as ex:
            st.error(f"Webhook failed: {ex}")

st.subheader("Detail + Chart")
detail = st.selectbox("Symbol", list(results.keys()))
res = results[detail]

if res.error:
    st.error(res.error)
else:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Yahoo (zones)", f"{res.last:.2f}")
    if res.mexc_last is not None:
        basis = res.mexc_last - res.last
        c2.metric("MEXC Live", f"{res.mexc_last:.2f}", delta=f"{basis:+.2f} vs Yahoo")
    else:
        c2.metric("MEXC Live", "N/A")
    c3.metric("ATR", f"{res.atr_d:.2f}")
    c4.metric("SMA50", f"{res.sma50:.2f}", delta="rising" if res.sma50_rising else "falling")
    c5.metric("Nearest Zone", f"{res.nearest_zone_dist_atr:.2f} ATR")
    c6.metric("Bias", res.market_bias)

    fig = make_chart(res)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Reliable external sources**")
    yf_news = f"https://finance.yahoo.com/quote/{res.equity}/news"
    yf_chart = f"https://finance.yahoo.com/chart/{res.equity}"
    st.markdown(f"- [Yahoo Finance News — {res.equity}]({yf_news})")
    st.markdown(f"- [Yahoo Finance Chart — {res.equity}]({yf_chart})")
    st.markdown(f"- [TradingView — {res.equity}](https://www.tradingview.com/chart/?symbol={res.equity})")

    if res.entries:
        st.markdown("**High-Quality Entries** (with position size from sidebar inputs)")
        for e in res.entries:
            ps = calc_position_size(account_size, risk_pct, e.entry, e.stop)
            st.success(
                f"**{e.side.upper()}** Q={e.quality} | Entry `{e.entry}` | Stop `{e.stop}` | Target `{e.target}` | "
                f"R:R **{e.rr}** | Dist {e.dist_atr} ATR  \n"
                f"{e.reason}  \n"
                f"**Suggested size:** {ps['size_coins']} coins  |  Notional ≈ {ps['notional']} USDT  |  "
                f"Risk amount ≈ {ps['risk_amount']} USDT ({risk_pct}% of account)"
            )
    else:
        st.info("No setups currently pass the maximum quality filters for this symbol.")

    with st.expander("All ranked zones"):
        zdata = [{"TF": z.tf, "Src": z.source, "Kind": z.kind, "Center": round(z.center, 3),
                  "Low": round(z.low, 3), "High": round(z.high, 3), "Touches": z.touches,
                  "Strength": round(z.strength, 2), "Fib": "Y" if z.fib else ""} for z in res.zones]
        st.dataframe(pd.DataFrame(zdata), use_container_width=True, hide_index=True)

st.markdown("---")
st.caption(f"Last run: {st.session_state.last_run_v2} · Maximum practical filters applied · Expect far fewer but higher-quality signals · Not financial advice")
