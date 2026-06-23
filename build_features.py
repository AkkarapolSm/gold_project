"""สร้าง feature files สำหรับทุก Timeframe"""
import MetaTrader5 as mt5
from gold_mt5_pipeline import connect_mt5, fetch_candles
from gold_features import build_features
from pathlib import Path

Path("gold_data").mkdir(exist_ok=True)

if not connect_mt5():
    print("[ERROR] MT5 เชื่อมต่อไม่ได้")
    exit(1)

for tf_name, tf_val in [
    ("M15", mt5.TIMEFRAME_M15),
    ("M30", mt5.TIMEFRAME_M30),
    ("H1",  mt5.TIMEFRAME_H1),
]:
    print(f"\nBuilding features: {tf_name}...")
    df = fetch_candles(tf_name, tf_val, n=10000)
    if df.empty:
        print(f"  [SKIP] ดึงข้อมูลไม่ได้")
        continue
    df_feat = build_features(df, target_atr_mult=0.5)
    out = Path("gold_data") / f"XAUUSD_{tf_name}_features.csv"
    df_feat.to_csv(out)
    print(f"  ✅ {len(df_feat)} rows → {out}")

mt5.shutdown()
print("\n✅ เสร็จแล้ว! ตอนนี้รัน gold_model.py ได้เลย")
