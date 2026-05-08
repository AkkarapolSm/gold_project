"""
=============================================================
 Market Regime Detection
 แยก Trending / Sideways / Volatile — เลือก model ให้เหมาะสม
=============================================================
 4 regime:
   TRENDING_UP   → ใช้ trend-following model (EMA, MACD weight สูง)
   TRENDING_DOWN → ใช้ trend-following model
   SIDEWAYS      → ใช้ mean-reversion model (RSI, BB weight สูง)
   VOLATILE      → ลด position size หรือหยุดเทรด
=============================================================
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

DATA_DIR  = Path("./gold_data")
MODEL_DIR = Path("./gold_models")


# ════════════════════════════════════════════════════════════
#  REGIME INDICATORS
# ════════════════════════════════════════════════════════════

def detect_regime(df: pd.DataFrame, lookback: int = 50) -> dict:
    """
    ตรวจจับ market regime จาก indicators หลายตัว
    คืน regime label + confidence + แนะนำ model ที่เหมาะสม
    """
    if len(df) < lookback + 20:
        return {"regime": "UNKNOWN", "confidence": 0}

    window = df.iloc[-lookback:].copy()
    c = window["Close"]
    h = window["High"]
    l = window["Low"]

    scores = {}

    # ── 1. ADX (ความแรงของ trend) ────────────────────────────
    adx_val = _calc_adx(h, l, c, 14)
    adx_last = adx_val.iloc[-1] if not adx_val.empty else 20
    scores["adx"] = adx_last
    # ADX > 25 = trending, < 20 = sideways

    # ── 2. Linear Regression R² (linearity of price) ─────────
    prices  = c.values
    x       = np.arange(len(prices))
    slope, intercept, r_val, _, _ = linregress(x, prices)
    r2      = r_val ** 2
    scores["r2"]    = r2
    scores["slope"] = slope
    # R² > 0.7 = strong trend, < 0.3 = sideways

    # ── 3. Efficiency Ratio (Kaufman) ─────────────────────────
    direction = abs(c.iloc[-1] - c.iloc[0])
    volatility= (c.diff().abs()).sum()
    er        = direction / volatility if volatility > 0 else 0
    scores["er"] = er
    # ER > 0.5 = trending, < 0.2 = sideways/choppy

    # ── 4. Bollinger Band Width (ความผันผวน) ─────────────────
    bb_mid   = c.rolling(20).mean()
    bb_std   = c.rolling(20).std()
    bb_width = (bb_std * 4 / bb_mid * 100).iloc[-1]
    scores["bb_width"] = bb_width
    # width สูง = volatile, ต่ำ = quiet/sideways

    # ── 5. ATR ratio (short vs long term volatility) ──────────
    atr_short = _atr(h, l, c, 7).iloc[-1]
    atr_long  = _atr(h, l, c, 21).iloc[-1]
    atr_ratio = atr_short / atr_long if atr_long > 0 else 1.0
    scores["atr_ratio"] = atr_ratio
    # > 1.3 = volatility expanding, < 0.7 = contracting

    # ── 6. Price position relative to EMAs ────────────────────
    ema21  = c.ewm(span=21).mean().iloc[-1]
    ema50  = c.ewm(span=50).mean().iloc[-1]
    last_c = c.iloc[-1]
    above_emas = (last_c > ema21) and (last_c > ema50)
    below_emas = (last_c < ema21) and (last_c < ema50)

    # ── 7. Consecutive directional candles ────────────────────
    bull_streak = _streak(window["Close"], window["Open"])
    scores["bull_streak"] = bull_streak

    # ════════════════════════════════════════════════════════
    #  CLASSIFY REGIME
    # ════════════════════════════════════════════════════════

    # Volatile: ATR สูงผิดปกติ
    if atr_ratio > 1.4 and bb_width > 2.5:
        regime = "VOLATILE"
        confidence = min(int((atr_ratio - 1.0) * 60 + (bb_width - 1.5) * 10), 90)

    # Trending: ADX แรง + R² สูง + ER สูง
    elif adx_last > 25 and r2 > 0.55 and er > 0.4:
        if slope > 0 and above_emas:
            regime = "TRENDING_UP"
        elif slope < 0 and below_emas:
            regime = "TRENDING_DOWN"
        else:
            regime = "TRENDING_UP" if slope > 0 else "TRENDING_DOWN"

        trend_strength = (adx_last / 50) * 0.4 + r2 * 0.4 + er * 0.2
        confidence = min(int(trend_strength * 100), 92)

    # Sideways: ADX อ่อน + R² ต่ำ
    elif adx_last < 22 and r2 < 0.4 and er < 0.35:
        regime = "SIDEWAYS"
        sideways_strength = (1 - adx_last/50) * 0.4 + (1-r2) * 0.4 + (1-er) * 0.2
        confidence = min(int(sideways_strength * 100), 90)

    # Middle ground
    else:
        if adx_last > 22 and slope > 0:
            regime = "TRENDING_UP"
        elif adx_last > 22 and slope < 0:
            regime = "TRENDING_DOWN"
        else:
            regime = "SIDEWAYS"
        confidence = 45

    # ── แนะนำ model และ parameter ────────────────────────────
    model_config = _get_model_config(regime, scores)

    result = {
        "regime"       : regime,
        "confidence"   : confidence,
        "indicators"   : {
            "ADX"       : round(adx_last, 1),
            "R2"        : round(r2, 3),
            "eff_ratio" : round(er, 3),
            "bb_width"  : round(bb_width, 2),
            "atr_ratio" : round(atr_ratio, 3),
            "slope_dir" : "up" if slope > 0 else "down",
        },
        "model_config" : model_config,
        "updated"      : datetime.now(timezone.utc).isoformat(),
    }

    print(f"[REGIME] {regime} ({confidence}%) | "
          f"ADX={adx_last:.1f} R²={r2:.2f} ER={er:.2f}")

    return result


# ════════════════════════════════════════════════════════════
#  MODEL CONFIG PER REGIME
# ════════════════════════════════════════════════════════════

def _get_model_config(regime: str, scores: dict) -> dict:
    """
    แนะนำ configuration ของ Ensemble ที่เหมาะกับ regime นั้น
    """
    configs = {
        "TRENDING_UP": {
            "model"          : "ensemble_trend",
            "xgb_weight"     : 0.30,
            "lgb_weight"     : 0.30,
            "lstm_weight"    : 0.40,   # LSTM เก่ง trend
            "feature_focus"  : ["trend", "momentum", "lag"],
            "position_size"  : 1.0,    # full size
            "preferred_side" : "LONG",
            "note"           : "ตลาด uptrend — ให้น้ำหนัก LSTM + trend features",
        },
        "TRENDING_DOWN": {
            "model"          : "ensemble_trend",
            "xgb_weight"     : 0.30,
            "lgb_weight"     : 0.30,
            "lstm_weight"    : 0.40,
            "feature_focus"  : ["trend", "momentum", "lag"],
            "position_size"  : 1.0,
            "preferred_side" : "SHORT",
            "note"           : "ตลาด downtrend — ให้น้ำหนัก LSTM + trend features",
        },
        "SIDEWAYS": {
            "model"          : "ensemble_reversion",
            "xgb_weight"     : 0.45,   # XGB เก่ง range
            "lgb_weight"     : 0.45,
            "lstm_weight"    : 0.10,   # LSTM ไม่เหมาะกับ choppy
            "feature_focus"  : ["momentum", "volatility", "candle", "structure"],
            "position_size"  : 0.7,    # ลด size ลง 30%
            "preferred_side" : "BOTH",
            "note"           : "ตลาด sideways — ใช้ RSI/BB mean-reversion, ลด position",
        },
        "VOLATILE": {
            "model"          : "pause",
            "xgb_weight"     : 0.33,
            "lgb_weight"     : 0.33,
            "lstm_weight"    : 0.34,
            "feature_focus"  : ["volatility"],
            "position_size"  : 0.3,    # ลด size มาก
            "preferred_side" : "NONE",
            "note"           : "⚠️ ตลาดผันผวนสูง — แนะนำหยุดเทรด หรือ SL กว้างขึ้น",
        },
        "UNKNOWN": {
            "model"          : "ensemble_default",
            "xgb_weight"     : 0.35,
            "lgb_weight"     : 0.35,
            "lstm_weight"    : 0.30,
            "feature_focus"  : ["trend","momentum","volatility"],
            "position_size"  : 0.5,
            "preferred_side" : "BOTH",
            "note"           : "ไม่ทราบ regime — ลด position ครึ่งหนึ่ง",
        },
    }
    return configs.get(regime, configs["UNKNOWN"])


# ════════════════════════════════════════════════════════════
#  REGIME-AWARE PREDICTOR
# ════════════════════════════════════════════════════════════

def predict_with_regime(df: pd.DataFrame, tf: str,
                        last_close: float, atr: float) -> dict:
    """
    ตรวจ regime → โหลด model ที่เหมาะสม → predict → ปรับ position size
    """
    import joblib
    from gold_model import EnsembleModel

    regime_info = detect_regime(df)
    regime      = regime_info["regime"]
    config      = regime_info["model_config"]

    # ── โหลด Ensemble และปรับ weights ตาม regime ─────────────
    try:
        ens = EnsembleModel.load_all(tf)
        ens.weights = np.array([
            config["xgb_weight"],
            config["lgb_weight"],
            config["lstm_weight"],
        ])
        ens.weights /= ens.weights.sum()
    except Exception as e:
        print(f"[REGIME PREDICT] โหลด model ไม่ได้: {e}")
        return {"error": str(e), "regime": regime}

    # ── Predict ───────────────────────────────────────────────
    signal = ens.predict_signal(df, last_close, atr)

    # ── ปรับ position size ────────────────────────────────────
    pos_size = config["position_size"]
    signal["position_size"]   = pos_size
    signal["regime"]          = regime
    signal["regime_conf"]     = regime_info["confidence"]
    signal["regime_note"]     = config["note"]
    signal["preferred_side"]  = config["preferred_side"]

    # ปรับ TP/SL ถ้าเป็น volatile (SL กว้างขึ้น)
    if regime == "VOLATILE" and signal.get("SL") and signal.get("TP"):
        atr_mult = 2.0  # กว้างขึ้น
        if signal.get("direction","").startswith("UP"):
            signal["SL"] = round(last_close - atr * atr_mult, 2)
            signal["TP"] = round(last_close + atr * atr_mult * 1.5, 2)
        elif signal.get("direction","").startswith("DOWN"):
            signal["SL"] = round(last_close + atr * atr_mult, 2)
            signal["TP"] = round(last_close - atr * atr_mult * 1.5, 2)

    # ยกเลิก signal ถ้า sideways และ signal ไม่ใช่ทิศที่แนะนำ
    if (regime == "SIDEWAYS" and
        config["preferred_side"] == "BOTH" and
        signal.get("confidence")):
        conf = float(signal["confidence"].replace("%",""))
        if conf < 65:
            signal["direction"] = "NEUTRAL — (Sideways regime)"
            signal["entry"] = signal["SL"] = signal["TP"] = None

    print(f"[REGIME] {regime} → pos_size={pos_size} | {signal.get('direction','?')}")
    return signal


# ════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════

def _atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int) -> pd.Series:
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()

def _calc_adx(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> pd.Series:
    try:
        import ta
        adx = ta.trend.ADXIndicator(h, l, c, window=n)
        return adx.adx()
    except Exception:
        # fallback คำนวณเอง
        atr_s = _atr(h, l, c, n)
        up_move   = h.diff()
        down_move = -l.diff()
        pdm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        ndm = down_move.where((down_move > up_move) & (down_move > 0), 0)
        pdi = (pdm.ewm(span=n).mean() / atr_s * 100)
        ndi = (ndm.ewm(span=n).mean() / atr_s * 100)
        dx  = ((pdi - ndi).abs() / (pdi + ndi) * 100).fillna(0)
        return dx.ewm(span=n).mean()

def _streak(close: pd.Series, open_: pd.Series) -> int:
    """จำนวน candle bull ติดต่อกัน (ลบ = bear streak)"""
    bulls = (close > open_).astype(int).replace(0, -1).values
    streak, count = bulls[-1], 1
    for b in reversed(bulls[:-1]):
        if b == streak:
            count += 1
        else:
            break
    return count * streak


# ════════════════════════════════════════════════════════════
#  SAVE / LOAD REGIME
# ════════════════════════════════════════════════════════════

def save_regime(regime_info: dict, tf: str):
    path = DATA_DIR / f"regime_{tf}.json"
    with open(path, "w") as f:
        json.dump(regime_info, f, indent=2, default=str)

def load_regime(tf: str) -> dict:
    path = DATA_DIR / f"regime_{tf}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🔍 Regime Detection Demo\n")
    import MetaTrader5 as mt5
    from gold_mt5_pipeline import connect_mt5, fetch_candles, TIMEFRAMES
    from gold_features import build_features

    if not connect_mt5():
        print("MT5 ไม่ได้เปิด — ใช้ dummy data")
        import numpy as np
        dates = pd.date_range("2024-01-01", periods=300, freq="15min", tz="UTC")
        np.random.seed(42)
        p = 2000 + np.cumsum(np.random.randn(300) * 1.5)
        df = pd.DataFrame({
            "Open"  : p + np.random.randn(300)*0.3,
            "High"  : p + abs(np.random.randn(300))*1.5,
            "Low"   : p - abs(np.random.randn(300))*1.5,
            "Close" : p,
            "Volume": np.random.randint(100,1000,300).astype(float),
        }, index=dates)
        df["High"] = df[["Open","High","Close"]].max(axis=1)
        df["Low"]  = df[["Open","Low","Close"]].min(axis=1)
    else:
        df = fetch_candles("M15", TIMEFRAMES["M15"], n=300)
        mt5.shutdown()

    for lookback in [20, 50, 100]:
        print(f"\n── lookback={lookback} ──")
        regime = detect_regime(df, lookback=lookback)
        print(f"   Regime     : {regime['regime']} ({regime['confidence']}%)")
        print(f"   Indicators : {regime['indicators']}")
        print(f"   Note       : {regime['model_config']['note']}")
        print(f"   Pos size   : {regime['model_config']['position_size']*100:.0f}%")