"""
=============================================================
 Gold Price Data Pipeline — MT5 + Exness
=============================================================
 ติดตั้ง:
   pip install MetaTrader5 pandas numpy ta schedule python-dotenv

 วิธีใช้:
   1. สร้างไฟล์ .env ใส่ข้อมูล login:
      MT5_LOGIN=213596035
      MT5_PASSWORD=your_password
      MT5_SERVER=Exness-MT5Real
   2. เปิด MetaTrader 5 ของ Exness ให้ login อยู่ก่อน
   3. รัน: python gold_mt5_pipeline.py
=============================================================
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import ta
import time
import schedule
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Config ───────────────────────────────────────────────
SYMBOL         = "XAUUSDm"
TIMEFRAMES     = {
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1" : mt5.TIMEFRAME_H1,
}
CANDLE_HISTORY = 500
DATA_DIR       = Path("./gold_data")
DATA_DIR.mkdir(exist_ok=True)

# ─── MT5 Credentials (จาก .env) ───────────────────────────
MT5_LOGIN    = int(os.getenv("MT5_LOGIN", "0"))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER   = os.getenv("MT5_SERVER", "Exness-MT5Real")


# ─── 1. เชื่อมต่อ MT5 ─────────────────────────────────────
def connect_mt5() -> bool:
    global SYMBOL

    # ── ลองเชื่อมต่อแบบมี credentials ──────────────────────
    if MT5_LOGIN and MT5_PASSWORD:
        ok = mt5.initialize(
            login    = MT5_LOGIN,
            password = MT5_PASSWORD,
            server   = MT5_SERVER,
        )
    else:
        # fallback: เชื่อมกับ MT5 ที่เปิดอยู่แล้ว
        ok = mt5.initialize()

    if not ok:
        err = mt5.last_error()
        print(f"[ERROR] MT5 initialize ล้มเหลว: {err}")
        if err[0] == -6:
            print("  → Authorization failed: ตรวจสอบ MT5_LOGIN / MT5_PASSWORD / MT5_SERVER ใน .env")
            print(f"  → Server ปัจจุบัน: {MT5_SERVER}")
            print("  → ดูชื่อ server ถูกต้องจาก title bar MT5 เช่น 'Exness-MT5Real4'")
        return False

    info = mt5.terminal_info()
    print(f"[OK] เชื่อมต่อ MT5 สำเร็จ — {info.name} build {info.build}")

    # ตรวจสอบ symbol
    for sym_try in [SYMBOL, "XAUUSD", "XAUUSDm", "GOLD"]:
        sym = mt5.symbol_info(sym_try)
        if sym is not None:
            SYMBOL = sym_try
            if not sym.visible:
                mt5.symbol_select(SYMBOL, True)
            print(f"[OK] Symbol: {SYMBOL} | bid={sym.bid:.2f} ask={sym.ask:.2f}")
            return True

    print("[ERROR] ไม่พบ symbol XAUUSDm / XAUUSD / GOLD")
    return False


# ─── 2. ดึง Candle (OHLCV) ───────────────────────────────
def fetch_candles(timeframe_name: str, timeframe: int, n: int = CANDLE_HISTORY) -> pd.DataFrame:
    rates = mt5.copy_rates_from_pos(SYMBOL, timeframe, 0, n)
    if rates is None or len(rates) == 0:
        print(f"[ERROR] ดึง candle {timeframe_name} ไม่ได้: {mt5.last_error()}")
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "tick_volume": "Volume"
    })
    df = df[["time", "Open", "High", "Low", "Close", "Volume"]]
    df = df.set_index("time")
    return df


# ─── 3. ดึงราคา Tick ปัจจุบัน (realtime) ────────────────
def fetch_tick() -> dict:
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        return {}
    return {
        "time"  : datetime.fromtimestamp(tick.time, tz=timezone.utc).isoformat(),
        "bid"   : tick.bid,
        "ask"   : tick.ask,
        "mid"   : round((tick.bid + tick.ask) / 2, 3),
        "spread": round((tick.ask - tick.bid) * 10, 1),
    }


# ─── 4. คำนวณ Indicators ──────────────────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 50:
        return df

    c = df["Close"]
    h = df["High"]
    l = df["Low"]

    df["EMA9"]   = ta.trend.ema_indicator(c, window=9)
    df["EMA21"]  = ta.trend.ema_indicator(c, window=21)
    df["EMA50"]  = ta.trend.ema_indicator(c, window=50)
    df["EMA200"] = ta.trend.ema_indicator(c, window=200)

    macd = ta.trend.MACD(c, window_slow=26, window_fast=12, window_sign=9)
    df["MACD"]        = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_hist"]   = macd.macd_diff()

    df["RSI14"] = ta.momentum.rsi(c, window=14)

    stoch = ta.momentum.StochasticOscillator(h, l, c, window=14, smooth_window=3)
    df["Stoch_K"] = stoch.stoch()
    df["Stoch_D"] = stoch.stoch_signal()

    bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_mid"]   = bb.bollinger_mavg()
    df["BB_lower"] = bb.bollinger_lband()
    df["BB_width"] = bb.bollinger_wband()

    df["ATR14"] = ta.volatility.average_true_range(h, l, c, window=14)
    df["OBV"]   = ta.volume.on_balance_volume(c, df["Volume"])

    df["body"]       = abs(df["Close"] - df["Open"])
    df["upper_wick"] = df["High"] - df[["Close", "Open"]].max(axis=1)
    df["lower_wick"] = df[["Close", "Open"]].min(axis=1) - df["Low"]
    df["is_bull"]    = (df["Close"] > df["Open"]).astype(int)
    df["trend_ema"]  = (df["EMA9"] > df["EMA21"]).astype(int)

    return df


# ─── 5. สร้าง Signal อย่างง่าย (Rule-based) ─────────────
def simple_signal(df: pd.DataFrame, tf_name: str) -> dict:
    if df.empty or len(df) < 3:
        return {}

    last = df.iloc[-1]
    prev = df.iloc[-2]

    signals = []
    score   = 0

    rsi = last["RSI14"]
    if rsi < 35:
        signals.append("RSI oversold → UP"); score += 2
    elif rsi > 65:
        signals.append("RSI overbought → DOWN"); score -= 2

    if prev["MACD"] < prev["MACD_signal"] and last["MACD"] > last["MACD_signal"]:
        signals.append("MACD golden cross → UP"); score += 3
    elif prev["MACD"] > prev["MACD_signal"] and last["MACD"] < last["MACD_signal"]:
        signals.append("MACD death cross → DOWN"); score -= 3

    if last["EMA9"] > last["EMA21"] > last["EMA50"]:
        signals.append("EMA bullish stack → UP"); score += 2
    elif last["EMA9"] < last["EMA21"] < last["EMA50"]:
        signals.append("EMA bearish stack → DOWN"); score -= 2

    if last["Close"] < last["BB_lower"]:
        signals.append("ราคาแตะ BB lower → UP"); score += 1
    elif last["Close"] > last["BB_upper"]:
        signals.append("ราคาแตะ BB upper → DOWN"); score -= 1

    atr   = last["ATR14"]
    price = last["Close"]

    if score > 0:
        direction = "UP ↑"
        entry = round(price, 2)
        sl    = round(price - atr * 1.5, 2)
        tp    = round(price + atr * 2.5, 2)
        pips_est = round(atr * 2.5, 1)
    elif score < 0:
        direction = "DOWN ↓"
        entry = round(price, 2)
        sl    = round(price + atr * 1.5, 2)
        tp    = round(price - atr * 2.5, 2)
        pips_est = round(atr * 2.5, 1)
    else:
        direction = "NEUTRAL —"
        entry = sl = tp = pips_est = None

    confidence = min(abs(score) * 15, 90)

    return {
        "timeframe" : tf_name,
        "time"      : str(df.index[-1]),
        "close"     : round(price, 2),
        "direction" : direction,
        "score"     : score,
        "confidence": f"{confidence}%",
        "entry"     : entry,
        "SL"        : sl,
        "TP"        : tp,
        "pips_est"  : pips_est,
        "signals"   : signals,
        "ATR"       : round(atr, 2),
        "RSI"       : round(rsi, 1),
    }


# ─── 6. บันทึกข้อมูลลงไฟล์ ───────────────────────────────
def save_data(df: pd.DataFrame, tf_name: str):
    path = DATA_DIR / f"XAUUSD_{tf_name}.csv"
    df.to_csv(path)


# ─── 7. Pipeline หลัก ─────────────────────────────────────
def run_pipeline():
    now = datetime.now().strftime("%H:%M:%S")
    print(f"\n{'='*55}")
    print(f"  🕐 {now}  |  Gold Pipeline Update")
    print(f"{'='*55}")

    tick = fetch_tick()
    if tick:
        print(f"\n  💰 ราคาปัจจุบัน")
        print(f"     Bid: {tick['bid']}  |  Ask: {tick['ask']}")
        print(f"     Mid: {tick['mid']}  |  Spread: {tick['spread']} pips")

    results = []

    for tf_name, tf_val in TIMEFRAMES.items():
        df = fetch_candles(tf_name, tf_val)
        if df.empty:
            continue
        df = add_indicators(df)
        save_data(df, tf_name)

        sig = simple_signal(df, tf_name)
        if sig:
            results.append(sig)
            print(f"\n  ⏱  Timeframe: {tf_name}")
            print(f"     ทิศทาง   : {sig['direction']}  ({sig['confidence']})")
            print(f"     Entry    : {sig['entry']}")
            print(f"     SL       : {sig['SL']}  |  TP: {sig['TP']}")
            print(f"     Pips est : ~{sig['pips_est']}")
            print(f"     RSI      : {sig['RSI']}  |  ATR: {sig['ATR']}")
            if sig["signals"]:
                for s in sig["signals"]:
                    print(f"     • {s}")

    out_path = DATA_DIR / "latest_signals.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"tick": tick, "signals": results}, f,
                  ensure_ascii=False, indent=2, default=str)

    print(f"\n  ✅ บันทึกข้อมูลใน {DATA_DIR}/")


# ─── 8. Main Loop ─────────────────────────────────────────
def main():
    print("🚀 Gold MT5 Pipeline — Exness")
    print(f"   Login: {MT5_LOGIN or '(จาก MT5 ที่เปิดอยู่)'}")
    print(f"   Server: {MT5_SERVER}")
    print("   กำลังเชื่อมต่อ MetaTrader 5...")

    if not connect_mt5():
        print("\n[ERROR] ไม่สามารถเชื่อมต่อ MT5 ได้")
        print("  แก้ไข: สร้างไฟล์ .env ดังนี้:")
        print("    MT5_LOGIN=213596035")
        print("    MT5_PASSWORD=your_password_here")
        print("    MT5_SERVER=Exness-MT5Real")
        print("  (ดูชื่อ Server จาก title bar ของ MT5)")
        return

    run_pipeline()

    schedule.every(1).minutes.do(run_pipeline)

    print("\n⏳ รอ schedule... (Ctrl+C เพื่อหยุด)")
    try:
        while True:
            schedule.run_pending()
            time.sleep(10)
    except KeyboardInterrupt:
        print("\n🛑 หยุดแล้ว")
    finally:
        mt5.shutdown()
        print("✅ ปิดการเชื่อมต่อ MT5 แล้ว")


if __name__ == "__main__":
    main()