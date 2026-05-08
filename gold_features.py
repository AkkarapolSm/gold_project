"""
=============================================================
 Gold Feature Engineering — XAU/USD
=============================================================
 ต่อจาก gold_mt5_pipeline.py
 ใช้:
   from gold_features import build_features
   df = build_features(df)  # df มาจาก fetch_candles()

 ติดตั้ง:
   pip install pandas numpy ta scipy
=============================================================
"""

import pandas as pd
import numpy as np
import ta
from scipy.stats import linregress


# ════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════

def build_features(df: pd.DataFrame, target_pips: float = None) -> pd.DataFrame:
    """
    รับ DataFrame OHLCV แล้วคืน DataFrame พร้อม features ครบ
    target_pips: ถ้าใส่มา จะสร้าง column 'target' สำหรับ train ML ด้วย
    """
    df = df.copy()
    df = _trend_features(df)
    df = _momentum_features(df)
    df = _volatility_features(df)
    df = _volume_features(df)
    df = _candle_pattern_features(df)
    df = _price_structure_features(df)
    df = _time_features(df)
    df = _lag_features(df)

    if target_pips is not None:
        df = _create_target(df, target_pips)

    # ลบแถวที่ indicator ยังไม่ครบ (warmup period)
    df = df.dropna()
    return df


# ════════════════════════════════════════════════════════════
#  1. TREND FEATURES
# ════════════════════════════════════════════════════════════

def _trend_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["Close"]
    h = df["High"]
    l = df["Low"]

    # EMA หลายช่วง
    for p in [9, 21, 50, 100, 200]:
        df[f"EMA{p}"] = ta.trend.ema_indicator(c, window=p)

    # SMA
    for p in [20, 50, 200]:
        df[f"SMA{p}"] = ta.trend.sma_indicator(c, window=p)

    # ราคาเทียบกับ EMA (เป็น % เพื่อให้ scale ไม่ขึ้นกับราคา)
    df["dist_EMA9"]   = (c - df["EMA9"])   / df["EMA9"]   * 100
    df["dist_EMA21"]  = (c - df["EMA21"])  / df["EMA21"]  * 100
    df["dist_EMA50"]  = (c - df["EMA50"])  / df["EMA50"]  * 100
    df["dist_EMA200"] = (c - df["EMA200"]) / df["EMA200"] * 100

    # EMA slope (ทิศทาง)
    df["slope_EMA9"]  = df["EMA9"].diff(3)  / df["EMA9"]  * 100
    df["slope_EMA21"] = df["EMA21"].diff(3) / df["EMA21"] * 100

    # EMA stack signal
    df["ema_bull_stack"] = (
        (df["EMA9"] > df["EMA21"]) &
        (df["EMA21"] > df["EMA50"]) &
        (df["EMA50"] > df["EMA200"])
    ).astype(int)
    df["ema_bear_stack"] = (
        (df["EMA9"] < df["EMA21"]) &
        (df["EMA21"] < df["EMA50"]) &
        (df["EMA50"] < df["EMA200"])
    ).astype(int)

    # MACD
    macd = ta.trend.MACD(c, window_slow=26, window_fast=12, window_sign=9)
    df["MACD"]        = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_hist"]   = macd.macd_diff()
    df["MACD_hist_prev"] = df["MACD_hist"].shift(1)

    # MACD crossover (1 = golden cross, -1 = death cross, 0 = none)
    df["MACD_cross"] = 0
    df.loc[
        (df["MACD_hist"] > 0) & (df["MACD_hist_prev"] <= 0), "MACD_cross"
    ] = 1
    df.loc[
        (df["MACD_hist"] < 0) & (df["MACD_hist_prev"] >= 0), "MACD_cross"
    ] = -1

    # ADX (ความแรงของ trend)
    adx = ta.trend.ADXIndicator(h, l, c, window=14)
    df["ADX"]     = adx.adx()
    df["ADX_pos"] = adx.adx_pos()   # +DI
    df["ADX_neg"] = adx.adx_neg()   # -DI
    df["ADX_diff"]= df["ADX_pos"] - df["ADX_neg"]

    # Ichimoku
    ichi = ta.trend.IchimokuIndicator(h, l,
        window1=9, window2=26, window3=52)
    df["ichi_conv"]  = ichi.ichimoku_conversion_line()   # Tenkan
    df["ichi_base"]  = ichi.ichimoku_base_line()          # Kijun
    df["ichi_a"]     = ichi.ichimoku_a()                  # Senkou A
    df["ichi_b"]     = ichi.ichimoku_b()                  # Senkou B
    df["above_cloud"]= (c > df[["ichi_a","ichi_b"]].max(axis=1)).astype(int)
    df["below_cloud"]= (c < df[["ichi_a","ichi_b"]].min(axis=1)).astype(int)

    # Linear Regression slope 20 candle
    def lr_slope(series, n=20):
        slopes = [np.nan] * n
        arr = series.values
        for i in range(n, len(arr)):
            y = arr[i-n:i]
            x = np.arange(n)
            s, _, _, _, _ = linregress(x, y)
            slopes.append(s)
        return pd.Series(slopes, index=series.index)

    df["lr_slope20"] = lr_slope(c, 20)

    return df


# ════════════════════════════════════════════════════════════
#  2. MOMENTUM FEATURES
# ════════════════════════════════════════════════════════════

def _momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["Close"]
    h = df["High"]
    l = df["Low"]

    # RSI หลายช่วง
    df["RSI7"]  = ta.momentum.rsi(c, window=7)
    df["RSI14"] = ta.momentum.rsi(c, window=14)
    df["RSI21"] = ta.momentum.rsi(c, window=21)

    # RSI slope
    df["RSI14_slope"] = df["RSI14"].diff(3)

    # RSI zones
    df["RSI_oversold"]   = (df["RSI14"] < 30).astype(int)
    df["RSI_overbought"] = (df["RSI14"] > 70).astype(int)

    # Stochastic
    stoch = ta.momentum.StochasticOscillator(h, l, c, window=14, smooth_window=3)
    df["Stoch_K"] = stoch.stoch()
    df["Stoch_D"] = stoch.stoch_signal()
    df["Stoch_diff"] = df["Stoch_K"] - df["Stoch_D"]

    # Stochastic crossover
    df["stoch_cross_up"]  = (
        (df["Stoch_K"] > df["Stoch_D"]) &
        (df["Stoch_K"].shift(1) <= df["Stoch_D"].shift(1))
    ).astype(int)
    df["stoch_cross_down"] = (
        (df["Stoch_K"] < df["Stoch_D"]) &
        (df["Stoch_K"].shift(1) >= df["Stoch_D"].shift(1))
    ).astype(int)

    # CCI
    df["CCI20"] = ta.trend.cci(h, l, c, window=20)

    # Williams %R
    df["WilliamsR"] = ta.momentum.williams_r(h, l, c, lbp=14)

    # Rate of Change
    df["ROC5"]  = ta.momentum.roc(c, window=5)
    df["ROC10"] = ta.momentum.roc(c, window=10)
    df["ROC20"] = ta.momentum.roc(c, window=20)

    # Momentum
    df["MOM10"] = c - c.shift(10)

    # TSI (True Strength Index)
    df["TSI"] = ta.momentum.tsi(c, window_slow=25, window_fast=13)

    return df


# ════════════════════════════════════════════════════════════
#  3. VOLATILITY FEATURES
# ════════════════════════════════════════════════════════════

def _volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["Close"]
    h = df["High"]
    l = df["Low"]

    # ATR หลายช่วง
    df["ATR7"]  = ta.volatility.average_true_range(h, l, c, window=7)
    df["ATR14"] = ta.volatility.average_true_range(h, l, c, window=14)
    df["ATR21"] = ta.volatility.average_true_range(h, l, c, window=21)

    # ATR ratio (ความผันผวนเพิ่มขึ้นหรือลดลง)
    df["ATR_ratio"] = df["ATR7"] / df["ATR14"]

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
    df["BB_upper"]   = bb.bollinger_hband()
    df["BB_mid"]     = bb.bollinger_mavg()
    df["BB_lower"]   = bb.bollinger_lband()
    df["BB_width"]   = bb.bollinger_wband()   # ความกว้าง band
    df["BB_pct"]     = bb.bollinger_pband()   # ตำแหน่งใน band (0=lower, 1=upper)

    # BB signals
    df["BB_squeeze"] = (df["BB_width"] < df["BB_width"].rolling(20).quantile(0.2)).astype(int)
    df["price_above_BB_upper"] = (c > df["BB_upper"]).astype(int)
    df["price_below_BB_lower"] = (c < df["BB_lower"]).astype(int)

    # Keltner Channel
    kc = ta.volatility.KeltnerChannel(h, l, c, window=20)
    df["KC_upper"] = kc.keltner_channel_hband()
    df["KC_lower"] = kc.keltner_channel_lband()

    # Historical Volatility (20 candle)
    df["HV20"] = c.pct_change().rolling(20).std() * np.sqrt(252) * 100

    # Candle range
    df["candle_range"]     = h - l
    df["candle_range_pct"] = (h - l) / c * 100

    # True Range
    df["TR"] = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs()
    ], axis=1).max(axis=1)

    return df


# ════════════════════════════════════════════════════════════
#  4. VOLUME FEATURES
# ════════════════════════════════════════════════════════════

def _volume_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["Close"]
    v = df["Volume"]

    # OBV
    df["OBV"] = ta.volume.on_balance_volume(c, v)
    df["OBV_slope"] = df["OBV"].diff(5)

    # Volume EMA
    df["VOL_EMA20"] = ta.trend.ema_indicator(v.astype(float), window=20)
    df["VOL_ratio"] = v / df["VOL_EMA20"]   # > 1 = volume สูงกว่าปกติ
    df["VOL_spike"] = (df["VOL_ratio"] > 2.0).astype(int)

    # VWAP approximation (ใช้ typical price)
    typical = (c + df["High"] + df["Low"]) / 3
    df["VWAP"] = (typical * v).cumsum() / v.cumsum()
    df["dist_VWAP"] = (c - df["VWAP"]) / df["VWAP"] * 100

    # MFI (Money Flow Index)
    df["MFI14"] = ta.volume.money_flow_index(
        df["High"], df["Low"], c, v.astype(float), window=14)

    # CMF (Chaikin Money Flow)
    df["CMF20"] = ta.volume.chaikin_money_flow(
        df["High"], df["Low"], c, v.astype(float), window=20)

    return df


# ════════════════════════════════════════════════════════════
#  5. CANDLE PATTERN FEATURES
# ════════════════════════════════════════════════════════════

def _candle_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    o = df["Open"]
    h = df["High"]
    l = df["Low"]
    c = df["Close"]

    body       = (c - o).abs()
    upper_wick = h - pd.concat([c, o], axis=1).max(axis=1)
    lower_wick = pd.concat([c, o], axis=1).min(axis=1) - l
    full_range = h - l

    df["body"]        = body
    df["upper_wick"]  = upper_wick
    df["lower_wick"]  = lower_wick
    df["body_ratio"]  = body / full_range.replace(0, np.nan)
    df["is_bull"]     = (c > o).astype(int)

    # Doji (body เล็กมาก)
    df["doji"] = (body / full_range.replace(0, np.nan) < 0.1).astype(int)

    # Hammer / Shooting Star
    df["hammer"] = (
        (lower_wick > body * 2) &
        (upper_wick < body * 0.5) &
        (df["is_bull"] == 1)
    ).astype(int)

    df["shooting_star"] = (
        (upper_wick > body * 2) &
        (lower_wick < body * 0.5) &
        (df["is_bull"] == 0)
    ).astype(int)

    # Engulfing
    df["bull_engulf"] = (
        (df["is_bull"] == 1) &
        (df["is_bull"].shift(1) == 0) &
        (c > o.shift(1)) &
        (o < c.shift(1))
    ).astype(int)

    df["bear_engulf"] = (
        (df["is_bull"] == 0) &
        (df["is_bull"].shift(1) == 1) &
        (c < o.shift(1)) &
        (o > c.shift(1))
    ).astype(int)

    # Inside bar
    df["inside_bar"] = (
        (h < h.shift(1)) &
        (l > l.shift(1))
    ).astype(int)

    # 3 candles consecutive
    df["3_bull"] = (
        (df["is_bull"] == 1) &
        (df["is_bull"].shift(1) == 1) &
        (df["is_bull"].shift(2) == 1)
    ).astype(int)

    df["3_bear"] = (
        (df["is_bull"] == 0) &
        (df["is_bull"].shift(1) == 0) &
        (df["is_bull"].shift(2) == 0)
    ).astype(int)

    return df


# ════════════════════════════════════════════════════════════
#  6. PRICE STRUCTURE (Support / Resistance)
# ════════════════════════════════════════════════════════════

def _price_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["Close"]
    h = df["High"]
    l = df["Low"]

    # Highest / Lowest ใน n candle
    for n in [10, 20, 50]:
        df[f"highest_{n}"]  = h.rolling(n).max()
        df[f"lowest_{n}"]   = l.rolling(n).min()
        df[f"dist_high_{n}"]= (c - df[f"highest_{n}"]) / c * 100
        df[f"dist_low_{n}"] = (c - df[f"lowest_{n}"])  / c * 100

    # Pivot Points (Classic) — คำนวณจาก candle ก่อนหน้า
    prev_h = h.shift(1)
    prev_l = l.shift(1)
    prev_c = c.shift(1)

    pp = (prev_h + prev_l + prev_c) / 3
    df["pivot_PP"] = pp
    df["pivot_R1"] = 2 * pp - prev_l
    df["pivot_S1"] = 2 * pp - prev_h
    df["pivot_R2"] = pp + (prev_h - prev_l)
    df["pivot_S2"] = pp - (prev_h - prev_l)

    df["dist_pivot"] = (c - pp) / pp * 100

    # Breakout signal
    df["breakout_up"]   = (c > df["highest_20"].shift(1)).astype(int)
    df["breakout_down"] = (c < df["lowest_20"].shift(1)).astype(int)

    # Gap (ช่องว่างระหว่าง candle)
    df["gap_up"]   = (df["Open"] > h.shift(1)).astype(int)
    df["gap_down"]  = (df["Open"] < l.shift(1)).astype(int)

    return df


# ════════════════════════════════════════════════════════════
#  7. TIME FEATURES
# ════════════════════════════════════════════════════════════

def _time_features(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index

    if hasattr(idx, "hour"):
        df["hour"]         = idx.hour
        df["minute"]       = idx.minute
        df["day_of_week"]  = idx.dayofweek   # 0=Mon ... 4=Fri
        df["is_london"]    = ((idx.hour >= 7)  & (idx.hour < 16)).astype(int)
        df["is_ny"]        = ((idx.hour >= 13) & (idx.hour < 22)).astype(int)
        df["is_overlap"]   = ((idx.hour >= 13) & (idx.hour < 16)).astype(int)
        df["is_asian"]     = ((idx.hour >= 0)  & (idx.hour < 7)).astype(int)

        # Sine/Cosine encode เพื่อให้โมเดลรู้ว่า 23 กับ 0 ใกล้กัน
        df["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
        df["dow_sin"]  = np.sin(2 * np.pi * idx.dayofweek / 5)
        df["dow_cos"]  = np.cos(2 * np.pi * idx.dayofweek / 5)

    return df


# ════════════════════════════════════════════════════════════
#  8. LAG FEATURES (ค่าย้อนหลัง)
# ════════════════════════════════════════════════════════════

def _lag_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["Close"]

    # Return (% change)
    for n in [1, 2, 3, 5, 10]:
        df[f"ret_{n}"] = c.pct_change(n) * 100

    # Lag Close (ราคาก่อนหน้า)
    for n in [1, 2, 3]:
        df[f"lag_close_{n}"] = c.shift(n)

    # Lag RSI
    for n in [1, 2, 3]:
        df[f"lag_RSI14_{n}"] = df["RSI14"].shift(n)

    # Lag MACD_hist
    for n in [1, 2, 3]:
        df[f"lag_MACD_hist_{n}"] = df["MACD_hist"].shift(n)

    # Rolling stats
    for n in [5, 10, 20]:
        df[f"rolling_mean_{n}"] = c.rolling(n).mean()
        df[f"rolling_std_{n}"]  = c.rolling(n).std()
        df[f"rolling_max_{n}"]  = c.rolling(n).max()
        df[f"rolling_min_{n}"]  = c.rolling(n).min()

    return df


# ════════════════════════════════════════════════════════════
#  9. TARGET VARIABLE (สำหรับ Train ML)
# ════════════════════════════════════════════════════════════

def _create_target(df: pd.DataFrame, min_pips: float = 3.0) -> pd.DataFrame:
    """
    สร้าง target สำหรับ classification:
      1  = ราคาขึ้น >= min_pips ใน N candle ข้างหน้า
     -1  = ราคาลง >= min_pips ใน N candle ข้างหน้า
      0  = sideways

    min_pips: จำนวน point ขั้นต่ำ (XAU/USD 1 pip ≈ 0.10)
    """
    c = df["Close"]
    future_high = df["High"].shift(-5).rolling(5, min_periods=1).max().shift(-(5-1))
    future_low  = df["Low"].shift(-5).rolling(5, min_periods=1).min().shift(-(5-1))

    df["future_high"] = future_high
    df["future_low"]  = future_low

    df["target"] = 0
    df.loc[(future_high - c) >= min_pips, "target"] = 1
    df.loc[(c - future_low)  >= min_pips, "target"] = -1

    # target_binary (สำหรับ binary classification)
    df["target_binary"] = (df["target"] == 1).astype(int)

    # future return (สำหรับ regression)
    df["future_ret_5"] = c.shift(-5).pct_change(5).shift(4) * 100

    return df


# ════════════════════════════════════════════════════════════
#  FEATURE SELECTOR — เลือก feature groups สำหรับ ML
# ════════════════════════════════════════════════════════════

def get_feature_columns(df: pd.DataFrame, groups: list = None) -> list:
    """
    คืน list ของ column ที่จะใช้ train
    groups: ['trend','momentum','volatility','volume','candle','structure','time','lag']
    ถ้าไม่ระบุ = ใช้ทุก group
    """
    all_groups = {
        "trend"      : ["EMA9","EMA21","EMA50","EMA200",
                        "dist_EMA9","dist_EMA21","dist_EMA50","dist_EMA200",
                        "slope_EMA9","slope_EMA21","ema_bull_stack","ema_bear_stack",
                        "MACD","MACD_signal","MACD_hist","MACD_cross",
                        "ADX","ADX_diff","lr_slope20",
                        "above_cloud","below_cloud"],
        "momentum"   : ["RSI7","RSI14","RSI21","RSI14_slope",
                        "RSI_oversold","RSI_overbought",
                        "Stoch_K","Stoch_D","Stoch_diff",
                        "stoch_cross_up","stoch_cross_down",
                        "CCI20","WilliamsR","ROC5","ROC10","MOM10","TSI"],
        "volatility" : ["ATR7","ATR14","ATR21","ATR_ratio",
                        "BB_width","BB_pct","BB_squeeze",
                        "price_above_BB_upper","price_below_BB_lower",
                        "HV20","candle_range_pct"],
        "volume"     : ["OBV_slope","VOL_ratio","VOL_spike",
                        "dist_VWAP","MFI14","CMF20"],
        "candle"     : ["body_ratio","doji","hammer","shooting_star",
                        "bull_engulf","bear_engulf","inside_bar",
                        "3_bull","3_bear","is_bull"],
        "structure"  : ["dist_high_10","dist_low_10",
                        "dist_high_20","dist_low_20",
                        "dist_pivot","breakout_up","breakout_down"],
        "time"       : ["hour_sin","hour_cos","dow_sin","dow_cos",
                        "is_london","is_ny","is_overlap","is_asian"],
        "lag"        : ["ret_1","ret_2","ret_3","ret_5",
                        "lag_RSI14_1","lag_RSI14_2",
                        "lag_MACD_hist_1","lag_MACD_hist_2",
                        "rolling_std_5","rolling_std_10","rolling_std_20"],
    }

    if groups is None:
        groups = list(all_groups.keys())

    selected = []
    for g in groups:
        cols = [c for c in all_groups.get(g, []) if c in df.columns]
        selected.extend(cols)

    return selected


# ════════════════════════════════════════════════════════════
#  DEMO — รันตรงๆ เพื่อทดสอบ
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import MetaTrader5 as mt5
    from gold_mt5_pipeline import connect_mt5, fetch_candles, TIMEFRAMES

    print("🔧 Gold Feature Engineering Demo\n")

    if not connect_mt5():
        print("MT5 ไม่ได้เปิด — ใช้ข้อมูล dummy แทน")

        # สร้าง dummy data สำหรับทดสอบ
        import pandas as pd
        import numpy as np
        dates = pd.date_range("2024-01-01", periods=300, freq="15min", tz="UTC")
        np.random.seed(42)
        price = 2000 + np.cumsum(np.random.randn(300) * 2)
        df_demo = pd.DataFrame({
            "Open"  : price + np.random.randn(300) * 0.5,
            "High"  : price + abs(np.random.randn(300)) * 2,
            "Low"   : price - abs(np.random.randn(300)) * 2,
            "Close" : price,
            "Volume": np.random.randint(100, 1000, 300).astype(float),
        }, index=dates)
        df_demo["High"] = df_demo[["Open","High","Close"]].max(axis=1)
        df_demo["Low"]  = df_demo[["Open","Low","Close"]].min(axis=1)

        df_feat = build_features(df_demo, target_pips=3.0)
    else:
        tf_name, tf_val = "M15", TIMEFRAMES["M15"]
        df_raw = fetch_candles(tf_name, tf_val, n=500)
        df_feat = build_features(df_raw, target_pips=3.0)
        mt5.shutdown()

    # สรุปผล
    feat_cols = get_feature_columns(df_feat)
    print(f"✅ Features ทั้งหมด : {len(feat_cols)} columns")
    print(f"✅ แถวข้อมูลที่ใช้ได้: {len(df_feat)} แถว")
    print(f"\nตัวอย่าง features 5 แถวล่าสุด:")
    preview_cols = ["Close","RSI14","MACD_hist","ATR14","BB_pct",
                    "ema_bull_stack","dist_EMA21","VOL_ratio","target"]
    preview_cols = [c for c in preview_cols if c in df_feat.columns]
    print(df_feat[preview_cols].tail(5).round(3).to_string())

    if "target" in df_feat.columns:
        dist = df_feat["target"].value_counts()
        print(f"\nTarget distribution:")
        print(f"  UP (1)      : {dist.get(1, 0)} ({dist.get(1,0)/len(df_feat)*100:.1f}%)")
        print(f"  DOWN (-1)   : {dist.get(-1,0)} ({dist.get(-1,0)/len(df_feat)*100:.1f}%)")
        print(f"  NEUTRAL (0) : {dist.get(0, 0)} ({dist.get(0,0)/len(df_feat)*100:.1f}%)")

    # บันทึกลงไฟล์
    df_feat.to_csv("gold_data/XAUUSD_M15_features.csv")
    print("\n💾 บันทึกแล้วที่ gold_data/XAUUSD_M15_features.csv")