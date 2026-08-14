#!/usr/bin/env python3
"""
MEXC Stock Futures S/R Dashboard (Streamlit)
============================================
Browser-based, always-ready trading tool for MEXC USDT-M Stock Futures.

Features implemented:
1. Expanded universe (core tech + mega-cap + growth + major indices/ETFs + more liquid names)
2. Multi-timeframe (Daily + 4H confluence)
3. Volume Profile / High Volume Nodes (HVN) as additional S/R
4. Bounce + Breakout / Retest entry logic
6. Interactive Streamlit web UI (open in browser and leave running)

Run:
  streamlit run mexc_sr_dashboard.py --server.port 8501
Then open http://localhost:8501 in your browser.
"""

from __future__ import annotations

import json
import math
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks
import streamlit as st

# ---------------------------------------------------------------------------
# Expanded Universe (MEXC Stock Futures → Yahoo equity/ETF ticker)
# ---------------------------------------------------------------------------

UNIVERSE: Dict[str, str] = {
    # Semiconductors / hardware
    "NVDAUSDT": "NVDA",
    "AMDUSDT": "AMD",
    "MUUSDT": "MU",
    "AVGOUSDT": "AVGO",
    "MRVLUSDT": "MRVL",
    "ASMLUSDT": "ASML",
    "ARMUSDT": "ARM",
    "QCOMUSDT": "QCOM",
    "INTCUSDT": "INTC",
    "SMCIUSDT": "SMCI",
    "SNDKUSDT": "SNDK",
    "LITEUSDT": "LITE",
    "AAOIUSDT": "AAOI",
    # Mega-cap & growth
    "TSLAUSDT": "TSLA",
    "AAPLUSDT": "AAPL",
    "MSFTUSDT": "MSFT",
    "METAUSDT": "META",
    "AMZNUSDT": "AMZN",
    "GOOGLUSDT": "GOOGL",
    "PLTRUSDT": "PLTR",
    "NFLXUSDT": "NFLX",
    "ORCLUSDT": "ORCL",
    "ADBEUSDT": "ADBE",
    "CSCOUSDT": "CSCO",
    "UBERUSDT": "UBER",
    "APPUSDT": "APP",
    "COINUSDT": "COIN",
    "HOODUSDT": "HOOD",
    "MSTRUSDT": "MSTR",
    "CRCLUSDT": "CRCL",
    # Additional liquid names
    "LLYUSDT": "LLY",
    "GEUSDT": "GE",
    "MCDUSDT": "MCD",
    "PEPUSDT": "PEP",
    "UNHUSDT": "UNH",
    "VUSDT": "V",
    "MAUSDT": "MA",
    # Indices / ETFs (map to liquid proxies)
    "SP500USDT": "SPY",
    "SPYUSDT": "SPY",
    "NAS100USDT": "QQQ",
    "QQQUSDT": "QQQ",
    "US30USDT": "DIA",
    "SOXLUSDT": "SOXL",
}

# Parameters
ATR_PERIOD = 14
PEAK_DISTANCE_D = 5
PEAK_DISTANCE_4H = 8
PROMINENCE_ATR_MULT = 0.75
ZONE_EPSILON_PCT = 0.0045
ZONE_EPSILON_ATR_MULT = 0.30
MIN_TOUCHES = 2
RECENCY_HALFLIFE = 45
VOLUME_WEIGHT = 0.40
FIB_LEVELS = [0.382, 0.5, 0.618]
FIB_TOL = 0.007
ATR_STOP_MULT = 0.55
MIN_RR = 1.8
VP_BINS = 48          # volume profile bins
HVN_PCTILE = 75       # high volume node threshold


# ---------------------------------------------------------------------------
# Data classes
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
    source: str = "swing"          # "swing" | "hvn" | "both"
    last_touch_idx: int = 0
    fib: bool = False
    tf: str = "1D"                 # "1D" | "4H" | "MTF"


@dataclass
class EntryIdea:
    side: str
    entry: float
    stop: float
    target: float
    rr: float
    reason: str
    zone_center: float
    style: str                     # "bounce" | "breakout" | "retest"


@dataclass
class AnalysisResult:
    futures: str
    equity: str
    last: float
    atr_d: float
    atr_4h: float
    zones: List[Zone] = field(default_factory=list)
    entries: List[EntryIdea] = field(default_factory=list)
    df_d: Optional[pd.DataFrame] = None
    df_4h: Optional[pd.DataFrame] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def fetch_ohlcv(ticker: str, range_: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval={interval}&range={range_}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=18) as r:
        data = json.loads(r.read())
    res = data["chart"]["result"][0]
    ts = res["timestamp"]
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame(
        {"Open": q["open"], "High": q["high"], "Low": q["low"], "Close": q["close"], "Volume": q["volume"]},
        index=pd.to_datetime(ts, unit="s"),
    )
    df = df.dropna()
    df = df[df["Volume"] > 0]
    return df


def to_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Resample 1h bars to 4h."""
    ohlc = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    return df_1h.resample("4h").agg(ohlc).dropna()


def atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev = close.shift(1)
    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


# ---------------------------------------------------------------------------
# Volume Profile / HVN
# ---------------------------------------------------------------------------

def volume_profile_hvn(df: pd.DataFrame, bins: int = VP_BINS, pctile: float = HVN_PCTILE) -> List[Tuple[float, float, float]]:
    """
    Returns list of (low, high, volume) for High Volume Nodes.
    Simple fixed-range VP on the lookback window.
    """
    if len(df) < 20:
        return []
    price_min = df["Low"].min()
    price_max = df["High"].max()
    if price_max <= price_min:
        return []
    bin_edges = np.linspace(price_min, price_max, bins + 1)
    vol_in_bin = np.zeros(bins)

    for _, row in df.iterrows():
        # distribute volume across the bar's range
        lo, hi, v = row["Low"], row["High"], row["Volume"]
        if hi <= lo:
            # single bin
            idx = np.searchsorted(bin_edges, lo, side="right") - 1
            idx = max(0, min(bins - 1, idx))
            vol_in_bin[idx] += v
            continue
        # proportional allocation
        for i in range(bins):
            b_lo, b_hi = bin_edges[i], bin_edges[i + 1]
            overlap = max(0.0, min(hi, b_hi) - max(lo, b_lo))
            if overlap > 0:
                vol_in_bin[i] += v * (overlap / (hi - lo))

    threshold = np.percentile(vol_in_bin[vol_in_bin > 0], pctile) if np.any(vol_in_bin > 0) else 0
    hvns = []
    i = 0
    while i < bins:
        if vol_in_bin[i] >= threshold and vol_in_bin[i] > 0:
            j = i
            while j + 1 < bins and vol_in_bin[j + 1] >= threshold:
                j += 1
            low = float(bin_edges[i])
            high = float(bin_edges[j + 1])
            total_v = float(vol_in_bin[i : j + 1].sum())
            hvns.append((low, high, total_v))
            i = j + 1
        else:
            i += 1
    return hvns


# ---------------------------------------------------------------------------
# Swing + Zone logic (multi-TF)
# ---------------------------------------------------------------------------

def find_swings(df: pd.DataFrame, atr_s: pd.Series, distance: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    highs = df["High"].values
    lows = df["Low"].values
    med_atr = float(np.nanmedian(atr_s.values[-80:])) if len(atr_s) > 20 else float(np.nanmedian(atr_s))
    prom = max(med_atr * PROMINENCE_ATR_MULT, 1e-8)
    h_idx, _ = find_peaks(highs, distance=distance, prominence=prom)
    l_idx, _ = find_peaks(-lows, distance=distance, prominence=prom)
    return h_idx, highs[h_idx], l_idx, lows[l_idx]


def cluster(prices: np.ndarray, vols: np.ndarray, idxs: np.ndarray, eps: float, kind: str, tf: str) -> List[Zone]:
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
    zones = []
    for cl in clusters:
        p, v, ix = prices[cl], vols[cl], idxs[cl]
        center = float(np.average(p, weights=v + 1e-9))
        low, high = float(p.min()), float(p.max())
        if high - low < 1e-9:
            half = eps * 0.4
            low -= half
            high += half
        zones.append(Zone(low=low, high=high, center=center, touches=len(cl),
                          total_volume=float(v.sum()), strength=0.0, kind=kind,
                          source="swing", last_touch_idx=int(ix.max()), tf=tf))
    return zones


def score_zones(zones: List[Zone], df: pd.DataFrame, sig_swing: Optional[Tuple[float, float]]) -> List[Zone]:
    n = len(df)
    vol_med = float(df["Volume"].median()) + 1e-9
    out = []
    for z in zones:
        age = n - 1 - z.last_touch_idx
        rec = math.exp(-math.log(2) * age / RECENCY_HALFLIFE)
        vol_f = 1.0 + VOLUME_WEIGHT * math.log1p(z.total_volume / vol_med)
        base = z.touches * vol_f * rec
        fib_b = 0.0
        if sig_swing:
            lo, hi = sig_swing
            rng = hi - lo
            if rng > 0:
                for f in FIB_LEVELS:
                    for level in (hi - f * rng, lo + f * rng):
                        if abs(z.center - level) / max(z.center, 1e-9) < FIB_TOL:
                            fib_b = 1.8
                            z.fib = True
                            break
        z.strength = base + fib_b
        if z.source == "hvn":
            z.strength *= 1.25   # slight boost for volume nodes
        out.append(z)
    out = [z for z in out if z.touches >= MIN_TOUCHES or z.strength > 1.8 or z.source == "hvn"]
    out.sort(key=lambda x: -x.strength)
    return out


def merge_close(zones: List[Zone], tol: float = 0.004) -> List[Zone]:
    if not zones:
        return []
    zones = sorted(zones, key=lambda z: z.center)
    merged = [zones[0]]
    for z in zones[1:]:
        last = merged[-1]
        if abs(z.center - last.center) / last.center < tol:
            new_low = min(last.low, z.low)
            new_high = max(last.high, z.high)
            new_vol = last.total_volume + z.total_volume
            new_touches = last.touches + z.touches
            new_c = (last.center * last.total_volume + z.center * z.total_volume) / (new_vol + 1e-9)
            kind = "both" if last.kind != z.kind else last.kind
            src = "both" if last.source != z.source else last.source
            merged[-1] = Zone(low=new_low, high=new_high, center=new_c, touches=new_touches,
                              total_volume=new_vol, strength=max(last.strength, z.strength),
                              kind=kind, source=src, last_touch_idx=max(last.last_touch_idx, z.last_touch_idx),
                              fib=last.fib or z.fib, tf="MTF" if last.tf != z.tf else last.tf)
        else:
            merged.append(z)
    return merged


# ---------------------------------------------------------------------------
# Entry generation (bounce + breakout + retest)
# ---------------------------------------------------------------------------

def generate_entries(zones: List[Zone], last: float, atr: float, df: pd.DataFrame) -> List[EntryIdea]:
    ideas = []
    atr_buf = atr * ATR_STOP_MULT
    strong = [z for z in zones if z.strength >= 2.0][:10]
    recent_close = df["Close"].iloc[-1]
    recent_vol = df["Volume"].iloc[-1]
    avg_vol = df["Volume"].iloc[-20:].mean()

    # --- Bounce ---
    for z in strong:
        if z.kind in ("support", "both") and last > z.high:
            if (last - z.high) < atr * 1.15:
                entry = z.high + atr * 0.12
                stop = z.low - atr_buf
                risk = entry - stop
                if risk <= 0:
                    continue
                higher = [zz for zz in strong if zz.center > entry + atr * 0.3]
                target = higher[0].low if higher else entry + risk * MIN_RR
                rr = (target - entry) / risk
                if rr >= MIN_RR * 0.85:
                    ideas.append(EntryIdea("long", round(entry, 4), round(stop, 4), round(target, 4),
                                           round(rr, 2), f"Bounce S {z.strength:.1f} ({z.source}/{z.tf})",
                                           z.center, "bounce"))

        if z.kind in ("resistance", "both") and last < z.low:
            if (z.low - last) < atr * 1.15:
                entry = z.low - atr * 0.12
                stop = z.high + atr_buf
                risk = stop - entry
                if risk <= 0:
                    continue
                lower = [zz for zz in strong if zz.center < entry - atr * 0.3]
                target = lower[0].high if lower else entry - risk * MIN_RR
                rr = (entry - target) / risk
                if rr >= MIN_RR * 0.85:
                    ideas.append(EntryIdea("short", round(entry, 4), round(stop, 4), round(target, 4),
                                           round(rr, 2), f"Bounce R {z.strength:.1f} ({z.source}/{z.tf})",
                                           z.center, "bounce"))

    # --- Breakout + simple retest detection ---
    # Look at last 3 bars for a close beyond a strong zone with volume
    if len(df) > 5:
        for z in strong:
            # Bullish breakout above resistance
            if z.kind in ("resistance", "both"):
                if (df["Close"].iloc[-2] < z.high and recent_close > z.high and
                        recent_vol > avg_vol * 1.3):
                    # potential retest entry near the broken level
                    entry = z.high + atr * 0.1
                    stop = z.low - atr_buf
                    risk = entry - stop
                    if risk > 0:
                        target = entry + risk * MIN_RR
                        ideas.append(EntryIdea("long", round(entry, 4), round(stop, 4), round(target, 4),
                                               round(MIN_RR, 2), f"Breakout+Retest R {z.strength:.1f}",
                                               z.center, "breakout"))

            # Bearish breakout below support
            if z.kind in ("support", "both"):
                if (df["Close"].iloc[-2] > z.low and recent_close < z.low and
                        recent_vol > avg_vol * 1.3):
                    entry = z.low - atr * 0.1
                    stop = z.high + atr_buf
                    risk = stop - entry
                    if risk > 0:
                        target = entry - risk * MIN_RR
                        ideas.append(EntryIdea("short", round(entry, 4), round(stop, 4), round(target, 4),
                                               round(MIN_RR, 2), f"Breakdown+Retest S {z.strength:.1f}",
                                               z.center, "breakout"))

    ideas.sort(key=lambda e: (-e.rr, -e.zone_center))
    # dedup roughly
    seen = set()
    unique = []
    for e in ideas:
        key = (e.side, round(e.entry, 2))
        if key not in seen:
            seen.add(key)
            unique.append(e)
    return unique[:5]


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

def analyze(futures: str, equity: str) -> AnalysisResult:
    try:
        df_d = fetch_ohlcv(equity, "6mo", "1d")
        if len(df_d) < 60:
            return AnalysisResult(futures, equity, 0, 0, 0, error="Insufficient daily data")

        # 4H via 1h resample (fetch longer range for enough bars)
        df_1h = fetch_ohlcv(equity, "3mo", "1h")
        df_4h = to_4h(df_1h) if len(df_1h) > 30 else None

        atr_d_s = atr(df_d)
        atr_d = float(atr_d_s.iloc[-1])
        last = float(df_d["Close"].iloc[-1])

        # Daily swings
        h_idx, h_p, l_idx, l_p = find_swings(df_d, atr_d_s, PEAK_DISTANCE_D)
        vol = df_d["Volume"].values
        eps_d = max(last * ZONE_EPSILON_PCT, atr_d * ZONE_EPSILON_ATR_MULT)
        zones_d = cluster(h_p, vol[h_idx], h_idx, eps_d, "resistance", "1D") + \
                  cluster(l_p, vol[l_idx], l_idx, eps_d, "support", "1D")

        # HVN from daily
        for lo, hi, v in volume_profile_hvn(df_d.iloc[-90:]):
            center = (lo + hi) / 2
            zones_d.append(Zone(low=lo, high=hi, center=center, touches=3, total_volume=v,
                                strength=0.0, kind="both", source="hvn", last_touch_idx=len(df_d)-5, tf="1D"))

        # Significant swing for Fib
        recent = df_d.iloc[-100:]
        sig = (float(recent["Low"].min()), float(recent["High"].max()))

        zones_d = score_zones(zones_d, df_d, sig)

        # 4H layer
        zones_4h = []
        atr_4h = atr_d
        if df_4h is not None and len(df_4h) > 40:
            atr_4h_s = atr(df_4h)
            atr_4h = float(atr_4h_s.iloc[-1])
            h4, hp4, l4, lp4 = find_swings(df_4h, atr_4h_s, PEAK_DISTANCE_4H)
            vol4 = df_4h["Volume"].values
            eps4 = max(last * ZONE_EPSILON_PCT * 0.9, atr_4h * ZONE_EPSILON_ATR_MULT)
            zones_4h = cluster(hp4, vol4[h4], h4, eps4, "resistance", "4H") + \
                       cluster(lp4, vol4[l4], l4, eps4, "support", "4H")
            zones_4h = score_zones(zones_4h, df_4h, sig)

        # Merge multi-TF
        all_z = merge_close(zones_d + zones_4h)
        all_z = sorted(all_z, key=lambda z: -z.strength)[:14]

        entries = generate_entries(all_z, last, atr_d, df_d)

        return AnalysisResult(futures, equity, last, atr_d, atr_4h, all_z, entries, df_d, df_4h)
    except Exception as e:
        return AnalysisResult(futures, equity, 0, 0, 0, error=str(e))


# ---------------------------------------------------------------------------
# Plotly chart
# ---------------------------------------------------------------------------

def make_chart(res: AnalysisResult) -> go.Figure:
    df = res.df_d
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25],
                        vertical_spacing=0.03)

    # Candles
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="OHLC", increasing_line_color="#26a69a", decreasing_line_color="#ef5350"
    ), row=1, col=1)

    colors = {"support": "rgba(38,166,154,0.25)", "resistance": "rgba(239,83,80,0.25)", "both": "rgba(156,39,176,0.22)"}
    line_c = {"support": "#26a69a", "resistance": "#ef5350", "both": "#9c27b0"}

    for z in res.zones[:9]:
        fig.add_hrect(y0=z.low, y1=z.high, fillcolor=colors.get(z.kind, "gray"),
                      line_width=0, row=1, col=1)
        fig.add_hline(y=z.center, line_dash="dot", line_color=line_c.get(z.kind, "gray"),
                      line_width=1, annotation_text=f"{z.kind[:1].upper()}{z.strength:.1f}",
                      annotation_position="right", row=1, col=1)

    # Entries
    for e in res.entries:
        color = "#26a69a" if e.side == "long" else "#ef5350"
        fig.add_trace(go.Scatter(
            x=[df.index[-1]], y=[e.entry], mode="markers",
            marker=dict(symbol="triangle-up" if e.side == "long" else "triangle-down",
                        size=14, color=color, line=dict(width=1, color="white")),
            name=f"{e.side.upper()} {e.style}",
            hovertext=f"{e.reason}<br>Entry {e.entry} Stop {e.stop} Tgt {e.target} R:R {e.rr}"
        ), row=1, col=1)
        fig.add_hline(y=e.stop, line_dash="dash", line_color=color, line_width=1, row=1, col=1)
        fig.add_hline(y=e.target, line_dash="dashdot", line_color=color, line_width=1, row=1, col=1)

    # Volume
    colors_vol = ["#26a69a" if c >= o else "#ef5350" for o, c in zip(df["Open"], df["Close"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=colors_vol, name="Vol", opacity=0.5),
                  row=2, col=1)

    fig.update_layout(
        title=f"{res.futures} ({res.equity})  |  Last {res.last:.2f}  ATR {res.atr_d:.2f}",
        xaxis_rangeslider_visible=False,
        height=680,
        template="plotly_dark",
        margin=dict(l=40, r=40, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="MEXC Stock Futures S/R", page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")

st.title("MEXC Stock Futures — Support / Resistance Dashboard")
st.caption("Multi-TF (Daily + 4H) · Volume Profile HVN · Bounce + Breakout/Retest · Always-on browser tool")

# Sidebar
with st.sidebar:
    st.header("Controls")
    all_syms = sorted(UNIVERSE.keys())
    default_sel = ["NVDAUSDT", "TSLAUSDT", "METAUSDT", "MUUSDT", "AAPLUSDT", "AMDUSDT",
                   "SP500USDT", "NAS100USDT", "PLTRUSDT", "HOODUSDT"]
    selected = st.multiselect("Symbols to analyze", all_syms, default=[s for s in default_sel if s in UNIVERSE])

    run_btn = st.button("🔄 Run / Refresh Analysis", type="primary", use_container_width=True)
    auto = st.checkbox("Auto-refresh every 5 min (keep tab open)", value=False)

    st.markdown("---")
    st.markdown("**Parameters** (advanced)")
    min_rr = st.slider("Min R:R", 1.5, 3.5, float(MIN_RR), 0.1)
    show_all_zones = st.checkbox("Show all zones (not just strong)", value=False)

    st.markdown("---")
    st.markdown(
        """
        **How to keep it always ready**
        1. Run `streamlit run mexc_sr_dashboard.py`
        2. Open the local URL in Chrome/Firefox
        3. Pin the tab or leave it open
        4. Enable auto-refresh if desired
        """
    )

# Main area
if "results" not in st.session_state:
    st.session_state.results = {}
    st.session_state.last_run = None

if run_btn or (auto and (st.session_state.last_run is None or
                         (datetime.now() - st.session_state.last_run).seconds > 300)):
    if not selected:
        st.warning("Select at least one symbol.")
    else:
        progress = st.progress(0.0, text="Fetching & analyzing…")
        results = {}
        for i, sym in enumerate(selected):
            equity = UNIVERSE[sym]
            progress.progress((i + 0.3) / len(selected), text=f"Analyzing {sym}…")
            res = analyze(sym, equity)
            results[sym] = res
            time.sleep(0.35)  # gentle rate limit
        st.session_state.results = results
        st.session_state.last_run = datetime.now()
        progress.progress(1.0, text="Done")
        time.sleep(0.3)
        progress.empty()

results = st.session_state.results

if not results:
    st.info("Select symbols in the sidebar and click **Run / Refresh Analysis**.")
    st.stop()

# Scanner table
st.subheader("Scanner — Actionable Setups")
rows = []
for sym, res in results.items():
    if res.error:
        rows.append({"Symbol": sym, "Last": "—", "Side": "ERROR", "Entry": res.error[:40], "R:R": "", "Style": "", "Reason": ""})
        continue
    if not res.entries:
        rows.append({"Symbol": sym, "Last": f"{res.last:.2f}", "Side": "—", "Entry": "No setup", "R:R": "", "Style": "", "Reason": ""})
        continue
    for e in res.entries[:2]:
        rows.append({
            "Symbol": sym,
            "Last": f"{res.last:.2f}",
            "Side": e.side.upper(),
            "Entry": f"{e.entry:.3f}",
            "Stop": f"{e.stop:.3f}",
            "Target": f"{e.target:.3f}",
            "R:R": e.rr,
            "Style": e.style,
            "Reason": e.reason,
        })

df_scan = pd.DataFrame(rows)
st.dataframe(df_scan, use_container_width=True, hide_index=True)

# Detail view
st.subheader("Detail View")
detail_sym = st.selectbox("Choose symbol for chart & zones", list(results.keys()))

res = results[detail_sym]
if res.error:
    st.error(res.error)
else:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Last", f"{res.last:.2f}")
    col2.metric("ATR (Daily)", f"{res.atr_d:.2f}")
    col3.metric("ATR (4H)", f"{res.atr_4h:.2f}")
    col4.metric("Zones", len(res.zones))

    # Chart
    fig = make_chart(res)
    st.plotly_chart(fig, use_container_width=True)

    # Zone table
    st.markdown("**Zones (strength ranked)**")
    zrows = []
    for z in res.zones:
        if not show_all_zones and z.strength < 1.8:
            continue
        zrows.append({
            "TF": z.tf,
            "Source": z.source,
            "Kind": z.kind,
            "Center": round(z.center, 3),
            "Low": round(z.low, 3),
            "High": round(z.high, 3),
            "Touches": z.touches,
            "Strength": round(z.strength, 2),
            "Fib": "Y" if z.fib else "",
        })
    st.dataframe(pd.DataFrame(zrows), use_container_width=True, hide_index=True)

    if res.entries:
        st.markdown("**Suggested Entries**")
        for e in res.entries:
            st.write(f"**{e.side.upper()}** ({e.style})  Entry `{e.entry}`  Stop `{e.stop}`  Target `{e.target}`  R:R **{e.rr}**  — {e.reason}")

st.markdown("---")
st.caption(f"Last run: {st.session_state.last_run} · Data via Yahoo Finance (underlying equities) · Futures track closely · Not financial advice")
