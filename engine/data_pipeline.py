"""
data_pipeline.py  (v7 — Multi-Timeframe + WaveTrend/StochRSI/MACD/MFI)
=========================================================================
Key fixes:
  1. Multi-timeframe data (1h, 4h, 1d) from Binance
  2. User's preferred indicators: WaveTrend, StochRSI, MACD, MFI, Volume
  3. PERCENTILE-based normalization (rank transform) — distribution-free
  4. Compact observation: single row of indicator values, NOT flattened window
"""

import os, time, logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
import requests

logger = logging.getLogger("data_pipeline")

BINANCE_BASE = "https://api.binance.com"


# ─── Data Fetching ──────────────────────────────────────────────────────────

def fetch_binance_klines(symbol, interval="1h", limit=1000, start=None):
    """Fetch klines from Binance public API."""
    all_rows = []
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if start:
        params["startTime"] = int(pd.Timestamp(start).timestamp() * 1000)

    end_ms = int(datetime.utcnow().timestamp() * 1000)
    current = params.get("startTime", end_ms - limit * 3600000)

    while current < end_ms:
        p = {"symbol": symbol, "interval": interval, "startTime": current,
             "endTime": end_ms, "limit": 1000}
        try:
            resp = requests.get(f"{BINANCE_BASE}/api/v3/klines", params=p, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"Binance error: {e}")
            break
        if not data:
            break
        all_rows.extend(data)
        current = data[-1][6] + 1
        time.sleep(0.15)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=[
        "openTime", "Open", "High", "Low", "Close", "Volume",
        "closeTime", "quoteVolume", "trades", "takerBuyBase", "takerBuyQuote", "ignore",
    ])
    df["Date"] = pd.to_datetime(df["openTime"], unit="ms")
    for col in ["Open", "High", "Low", "Close", "Volume", "quoteVolume"]:
        df[col] = df[col].astype(float)
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    df.set_index("Date", inplace=True)
    df = df[~df.index.duplicated(keep="last")]
    return df


def fetch_yahoo(ticker, start="2017-01-01", end=None):
    end = end or datetime.utcnow().strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    return df


# ─── Indicator Implementations ──────────────────────────────────────────────

def compute_wavetrend(df, n1=10, n2=21):
    """LazyBear WaveTrend — exact PineScript port."""
    ap = (df["High"] + df["Low"] + df["Close"]) / 3.0
    esa = ap.ewm(span=n1, adjust=False).mean()
    d = (ap - esa).abs().ewm(span=n1, adjust=False).mean()
    ci = (ap - esa) / (0.015 * d + 1e-10)
    wt1 = ci.ewm(span=n2, adjust=False).mean()
    wt2 = wt1.rolling(4).mean()
    return wt1, wt2


def compute_stoch_rsi(close, rsi_period=14, stoch_period=14, k_smooth=3, d_smooth=3):
    """Stochastic RSI."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).ewm(alpha=1/rsi_period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/rsi_period, adjust=False).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - 100 / (1 + rs)

    rsi_min = rsi.rolling(stoch_period).min()
    rsi_max = rsi.rolling(stoch_period).max()
    stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10) * 100
    k = stoch_rsi.rolling(k_smooth).mean()
    d = k.rolling(d_smooth).mean()
    return rsi, k, d


def compute_macd(close, fast=12, slow=26, signal=9):
    """MACD with histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_mfi(df, period=14):
    """Money Flow Index."""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    mf = tp * df["Volume"]
    pos_mf = mf.where(tp > tp.shift(1), 0.0).rolling(period).sum()
    neg_mf = mf.where(tp <= tp.shift(1), 0.0).rolling(period).sum()
    mfi = 100 - 100 / (1 + pos_mf / (neg_mf + 1e-10))
    return mfi


def compute_atr(df, period=14):
    """Average True Range."""
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands — returns %B position."""
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    pct_b = (close - lower) / (upper - lower + 1e-10)
    bandwidth = (upper - lower) / (sma + 1e-10)
    return pct_b, bandwidth


def compute_adx(df, period=14):
    """ADX with +DI and -DI."""
    h, l, c = df["High"], df["Low"], df["Close"]
    plus_dm = h.diff().where(h.diff() > l.diff().mul(-1), 0.0).clip(lower=0)
    minus_dm = (-l.diff()).where((-l.diff()) > h.diff(), 0.0).clip(lower=0)
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / (atr + 1e-10)
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / (atr + 1e-10)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, adjust=False).mean()
    return adx, plus_di, minus_di


def compute_obv(close, volume):
    """On-Balance Volume."""
    direction = np.sign(close.diff())
    return (direction * volume).cumsum()


# ─── Feature Builder ────────────────────────────────────────────────────────

def build_features(df, prefix=""):
    """Build all indicator features for a single timeframe DataFrame."""
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    feats = pd.DataFrame(index=df.index)

    # WaveTrend
    wt1, wt2 = compute_wavetrend(df)
    feats[f"{prefix}wt1"] = wt1
    feats[f"{prefix}wt2"] = wt2
    feats[f"{prefix}wt_cross"] = wt1 - wt2  # Cross signal
    feats[f"{prefix}wt_ob"] = (wt1 > 53).astype(float)  # Overbought
    feats[f"{prefix}wt_os"] = (wt1 < -53).astype(float)  # Oversold

    # StochRSI
    rsi, stoch_k, stoch_d = compute_stoch_rsi(c)
    feats[f"{prefix}rsi"] = rsi
    feats[f"{prefix}stoch_k"] = stoch_k
    feats[f"{prefix}stoch_d"] = stoch_d
    feats[f"{prefix}stoch_cross"] = stoch_k - stoch_d

    # MACD
    macd_line, macd_sig, macd_hist = compute_macd(c)
    feats[f"{prefix}macd"] = macd_line
    feats[f"{prefix}macd_sig"] = macd_sig
    feats[f"{prefix}macd_hist"] = macd_hist

    # MFI
    feats[f"{prefix}mfi"] = compute_mfi(df)

    # Bollinger
    bb_pct, bb_bw = compute_bollinger(c)
    feats[f"{prefix}bb_pct"] = bb_pct
    feats[f"{prefix}bb_bw"] = bb_bw

    # ADX
    adx, plus_di, minus_di = compute_adx(df)
    feats[f"{prefix}adx"] = adx
    feats[f"{prefix}di_diff"] = plus_di - minus_di

    # ATR (normalized by price)
    atr = compute_atr(df)
    feats[f"{prefix}atr_pct"] = atr / (c + 1e-10) * 100

    # Volume
    obv = compute_obv(c, v)
    feats[f"{prefix}vol_ratio"] = v / (v.rolling(20).mean() + 1e-10)
    feats[f"{prefix}obv_slope"] = obv.diff(5) / (obv.rolling(20).std() + 1e-10)

    # Price action
    feats[f"{prefix}ret_1"] = c.pct_change(1)
    feats[f"{prefix}ret_5"] = c.pct_change(5)
    feats[f"{prefix}ret_20"] = c.pct_change(20)
    feats[f"{prefix}hl_range"] = (h - l) / (c + 1e-10)

    return feats


# ─── Percentile Normalization ───────────────────────────────────────────────

def percentile_normalize(arr, window=200):
    """
    Rank-based normalization: each value is its percentile within a rolling window.
    Output is always in [0, 1] regardless of distribution — NO train/test shift.
    """
    result = np.zeros_like(arr)
    for i in range(len(arr)):
        start = max(0, i - window)
        window_data = arr[start:i+1]
        if len(window_data) < 5:
            result[i] = 0.5
        else:
            rank = np.searchsorted(np.sort(window_data), arr[i])
            result[i] = rank / len(window_data)
    return result


def normalize_features_percentile(features, window=200):
    """Apply percentile normalization to all features. Output in [-1, 1]."""
    result = np.zeros_like(features)
    for col in range(features.shape[1]):
        pct = percentile_normalize(features[:, col], window=window)
        result[:, col] = pct * 2.0 - 1.0  # Map [0,1] → [-1,1]
    return result.astype(np.float32)


# ─── Multi-Timeframe Pipeline ──────────────────────────────────────────────

def load_btc_multitf():
    """Load BTC data at 1h, 4h, 1d from Binance and compute features."""
    timeframes = {
        "1h": {"interval": "1h", "start": "2022-01-01"},
        "4h": {"interval": "4h", "start": "2020-01-01"},
        "1d": {"interval": "1d", "start": "2017-01-01"},
    }

    all_features = {}
    base_df = None

    for tf_name, cfg in timeframes.items():
        logger.info(f"  Fetching BTC {tf_name}...")
        df = fetch_binance_klines("BTCUSDT", interval=cfg["interval"], start=cfg["start"])
        if df.empty:
            logger.warning(f"  {tf_name} empty, trying Yahoo...")
            df = fetch_yahoo("BTC-USD", start=cfg["start"])
        if df.empty:
            logger.error(f"  No data for {tf_name}")
            continue

        logger.info(f"  {tf_name}: {len(df)} bars ({df.index[0]} → {df.index[-1]})")
        prefix = f"{tf_name}_" if tf_name != "1h" else ""
        feats = build_features(df, prefix=prefix)
        all_features[tf_name] = feats

        if tf_name == "1h":
            base_df = df.copy()

    if base_df is None or "1h" not in all_features:
        raise ValueError("Failed to load 1h base data")

    # Merge higher timeframes onto 1h base using forward-fill
    merged = all_features["1h"].copy()
    for tf_name in ["4h", "1d"]:
        if tf_name in all_features:
            htf = all_features[tf_name]
            # Reindex to 1h timestamps with forward fill
            htf_reindexed = htf.reindex(merged.index, method="ffill")
            merged = pd.concat([merged, htf_reindexed], axis=1)

    # Drop rows with NaN
    merged.dropna(inplace=True)

    # Align base_df
    base_df = base_df.loc[merged.index]

    logger.info(f"  Merged features: {merged.shape}")

    # Compute confluence score from indicators
    merged["confluence_long"] = (
        (merged.get("wt_cross", pd.Series(0, index=merged.index)) > 0).astype(float) +
        (merged.get("stoch_cross", pd.Series(0, index=merged.index)) > 0).astype(float) +
        (merged.get("macd_hist", pd.Series(0, index=merged.index)) > 0).astype(float) +
        (merged.get("rsi", pd.Series(50, index=merged.index)) < 40).astype(float) +
        (merged.get("mfi", pd.Series(50, index=merged.index)) < 40).astype(float)
    ) / 5.0

    merged["confluence_short"] = (
        (merged.get("wt_cross", pd.Series(0, index=merged.index)) < 0).astype(float) +
        (merged.get("stoch_cross", pd.Series(0, index=merged.index)) < 0).astype(float) +
        (merged.get("macd_hist", pd.Series(0, index=merged.index)) < 0).astype(float) +
        (merged.get("rsi", pd.Series(50, index=merged.index)) > 60).astype(float) +
        (merged.get("mfi", pd.Series(50, index=merged.index)) > 60).astype(float)
    ) / 5.0

    # Convert to numpy and normalize
    feature_names = list(merged.columns)
    raw_features = merged.values.astype(np.float64)

    # Replace any remaining NaN/Inf
    raw_features = np.nan_to_num(raw_features, nan=0.0, posinf=3.0, neginf=-3.0)

    # Percentile normalization — distribution-free!
    features = normalize_features_percentile(raw_features, window=200)

    prices = base_df["Close"].values.astype(np.float64)

    logger.info(f"  Final: features={features.shape}, prices={prices.shape}")
    return features, prices, feature_names


# ─── Legacy API for backward compatibility ──────────────────────────────────

ASSETS = {
    "BTCUSDT": {"yf": "BTC-USD", "binance": "BTCUSDT", "type": "crypto"},
}


def load_all_assets(start="2017-01-01", end=None):
    """Load BTC with multi-timeframe features."""
    features, prices, feat_names = load_btc_multitf()
    n = len(prices)
    # Create a dummy df for compatibility
    df = pd.DataFrame({
        "Close": prices,
        "Date": pd.date_range(start="2022-01-01", periods=n, freq="h"),
    })
    return {
        "BTCUSDT": {"df": df, "features": features, "feature_names": feat_names}
    }


FEATURE_COLS = []  # Not used in v7, kept for import compat
