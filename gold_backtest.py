"""
=============================================================
 Gold Strategy Backtest (D4) — ทดสอบกลยุทธ์จริงบนข้อมูลย้อนหลัง
=============================================================
 จำลองการเทรดจริงจากสัญญาณ ensemble:
   - เดินทีละแท่ง ใช้ proba ของ ensemble (XGB+LGBM+LSTM)
   - เปิดเมื่อ confidence >= threshold และไม่ใช่ NEUTRAL
   - SL/TP แบบ ATR เดียวกับ predict_signal (1.5 / 2.5 ATR)
   - ถือทีละ 1 ไม้ (flat→entry→exit) แล้วค่อยหาไม้ถัดไป
   - exit เมื่อชน SL/TP หรือครบ MAX_HOLD แท่ง (timeout, mark-to-market)
   - หักต้นทุน spread ต่อไม้

 ออก: equity curve, win-rate, profit factor, max drawdown, expectancy
 บันทึก gold_wfo/backtest_{tf}.csv (รายไม้) + backtest_{tf}.json (สรุป)

 รัน:
   python gold_backtest.py --tf M15 --conf 80
   python gold_backtest.py --tf H1  --conf 70 --tail 8000
   python gold_backtest.py --tf M15 --no-lstm        # เร็วขึ้น (XGB+LGB)
=============================================================
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gold_model import EnsembleModel, SEQ_LEN, DATA_DIR

WFO_DIR = Path("./gold_wfo")
WFO_DIR.mkdir(exist_ok=True)

# ATR multiplier (ต้องตรงกับ EnsembleModel.predict_signal)
ATR_SL = 1.5
ATR_TP = 2.5


def load_features(tf: str, tail: int = 0) -> pd.DataFrame:
    path = DATA_DIR / f"XAUUSD_{tf}_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"ไม่พบ {path} — รัน build_features.py ก่อน")
    df = pd.read_csv(path, index_col=0, parse_dates=True).dropna()
    if tail and len(df) > tail:
        df = df.iloc[-tail:]
    return df


def get_proba(ens: EnsembleModel, df: pd.DataFrame, use_lstm: bool = True):
    """คืน (proba[n,3], offset) — แถว j ตรงกับ df.iloc[offset+j]"""
    if use_lstm:
        try:
            p = ens.predict_proba_ensemble(df)
            return p, len(df) - len(p)
        except Exception as e:
            print(f"[BT] LSTM ensemble ล้มเหลว → fallback XGB+LGB ({e})")

    px = ens.xgb_m.predict_proba(df)
    pl = ens.lgb_m.predict_proba(df)
    w  = ens.weights
    ws = (w[0] + w[1]) or 1.0
    p  = (w[0] * px + w[1] * pl) / ws
    return p, 0


def run_backtest(tf: str = "M15", conf_min: float = 80.0,
                 lot: float = 0.01, spread: float = 0.30,
                 max_hold: int = 50, use_lstm: bool = True,
                 tail: int = 0) -> dict:
    df = load_features(tf, tail)
    if "ATR14" not in df.columns or "Close" not in df.columns:
        raise ValueError("feature CSV ขาด ATR14/Close")

    ens = EnsembleModel.load_all(tf)
    proba, offset = get_proba(ens, df, use_lstm)

    close = df["Close"].values
    high  = df["High"].values
    low   = df["Low"].values
    atr   = df["ATR14"].values
    times = df.index

    usd_per_price = 100.0 * lot   # 1.0 lot ทอง = $100 ต่อการเคลื่อน $1

    trades   = []
    i        = 0                  # index ใน proba
    n        = len(proba)

    while i < n:
        label = int(np.argmax(proba[i]))
        cf    = float(proba[i][label]) * 100
        if label == 1 or cf < conf_min:     # NEUTRAL หรือ conf ไม่พอ
            i += 1
            continue

        b      = offset + i                  # bar index ใน df
        is_long = (label == 2)
        entry  = close[b]
        a      = atr[b]
        if a <= 0:
            i += 1
            continue
        sl = entry - a * ATR_SL if is_long else entry + a * ATR_SL
        tp = entry + a * ATR_TP if is_long else entry - a * ATR_TP

        # เดินไปข้างหน้าหาผลลัพธ์
        exit_b, exit_px, outcome = None, None, None
        for j in range(b + 1, min(b + 1 + max_hold, len(close))):
            hi, lo = high[j], low[j]
            if is_long:
                hit_sl = lo <= sl
                hit_tp = hi >= tp
            else:
                hit_sl = hi >= sl
                hit_tp = lo <= tp
            if hit_sl and hit_tp:            # ชนทั้งคู่ในแท่งเดียว → ถือว่า SL ก่อน (อนุรักษ์)
                exit_b, exit_px, outcome = j, sl, "SL"; break
            if hit_sl:
                exit_b, exit_px, outcome = j, sl, "SL"; break
            if hit_tp:
                exit_b, exit_px, outcome = j, tp, "TP"; break

        if exit_b is None:                  # timeout → ปิดที่ราคาปิดแท่งสุดท้าย
            exit_b  = min(b + max_hold, len(close) - 1)
            exit_px = close[exit_b]
            outcome = "TIMEOUT"

        move   = (exit_px - entry) if is_long else (entry - exit_px)
        pnl_px = move - spread              # หักต้นทุน spread (price)
        pnl    = pnl_px * usd_per_price

        trades.append({
            "time"    : str(times[b])[:16],
            "dir"     : "UP" if is_long else "DOWN",
            "conf"    : round(cf, 1),
            "entry"   : round(entry, 2),
            "sl"      : round(sl, 2),
            "tp"      : round(tp, 2),
            "exit"    : round(exit_px, 2),
            "outcome" : outcome,
            "bars"    : exit_b - b,
            "pnl"     : round(pnl, 2),
        })

        # ต่อจากแท่งที่ปิด (ไม่เปิดซ้อน)
        i = (exit_b - offset) + 1

    return _summarize(tf, conf_min, lot, spread, max_hold, use_lstm, trades)


def _summarize(tf, conf_min, lot, spread, max_hold, use_lstm, trades) -> dict:
    n = len(trades)
    if n == 0:
        print(f"[BT] {tf}: ไม่มีไม้ที่ผ่านเกณฑ์ conf>={conf_min}%")
        return {"tf": tf, "trades": 0}

    pnls   = np.array([t["pnl"] for t in trades])
    wins   = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    equity = np.cumsum(pnls)

    peak    = np.maximum.accumulate(equity)
    drawdn  = peak - equity
    max_dd  = float(drawdn.max()) if n else 0.0

    gross_win  = float(wins.sum())
    gross_loss = float(-losses.sum())
    pf = (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)

    # max consecutive losses
    mcl = cur = 0
    for p in pnls:
        cur = cur + 1 if p < 0 else 0
        mcl = max(mcl, cur)

    summary = {
        "tf"           : tf,
        "conf_min"     : conf_min,
        "lot"          : lot,
        "spread"       : spread,
        "max_hold"     : max_hold,
        "use_lstm"     : use_lstm,
        "trades"       : n,
        "wins"         : int((pnls > 0).sum()),
        "losses"       : int((pnls < 0).sum()),
        "win_rate"     : round(float((pnls > 0).mean()) * 100, 1),
        "total_pnl"    : round(float(pnls.sum()), 2),
        "expectancy"   : round(float(pnls.mean()), 2),
        "avg_win"      : round(float(wins.mean()), 2) if len(wins) else 0.0,
        "avg_loss"     : round(float(losses.mean()), 2) if len(losses) else 0.0,
        "profit_factor": round(pf, 2) if pf != float("inf") else 999.0,
        "max_drawdown" : round(max_dd, 2),
        "max_consec_loss": mcl,
        "tp_hits"      : sum(1 for t in trades if t["outcome"] == "TP"),
        "sl_hits"      : sum(1 for t in trades if t["outcome"] == "SL"),
        "timeouts"     : sum(1 for t in trades if t["outcome"] == "TIMEOUT"),
    }

    # บันทึก
    pd.DataFrame(trades).assign(equity=np.round(equity, 2)) \
      .to_csv(WFO_DIR / f"backtest_{tf}.csv", index=False)
    (WFO_DIR / f"backtest_{tf}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # พิมพ์สรุป
    print(f"\n{'═'*56}")
    print(f"  Backtest {tf} | conf>={conf_min}% | lot {lot} | spread {spread} | LSTM={use_lstm}")
    print(f"{'═'*56}")
    print(f"  ไม้ทั้งหมด    : {summary['trades']}  (W {summary['wins']} / L {summary['losses']})")
    print(f"  Win rate     : {summary['win_rate']}%   (TP {summary['tp_hits']} / SL {summary['sl_hits']} / TO {summary['timeouts']})")
    print(f"  Total P/L    : {summary['total_pnl']:+.2f} USD")
    print(f"  Expectancy   : {summary['expectancy']:+.2f} USD/ไม้")
    print(f"  Avg win/loss : {summary['avg_win']:+.2f} / {summary['avg_loss']:+.2f}")
    print(f"  Profit factor: {summary['profit_factor']}")
    print(f"  Max drawdown : {summary['max_drawdown']:.2f} USD")
    print(f"  Max ขาดทุนติด : {summary['max_consec_loss']} ไม้")
    print(f"  💾 gold_wfo/backtest_{tf}.csv + .json")
    print(f"  ⚠️  in-sample: โมเดลอาจเทรนทับช่วงนี้ → ดู out-of-sample จาก gold_walk_forward.py")
    print(f"{'═'*56}\n")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Gold strategy backtest")
    ap.add_argument("--tf", default="M15", choices=["M15", "M30", "H1"])
    ap.add_argument("--conf", type=float, default=80.0, help="confidence ขั้นต่ำ เป็นเปอร์เซ็นต์")
    ap.add_argument("--lot", type=float, default=0.01)
    ap.add_argument("--spread", type=float, default=0.30, help="ต้นทุน spread ต่อไม้ (price)")
    ap.add_argument("--max-hold", type=int, default=50, help="ถือสูงสุดกี่แท่ง")
    ap.add_argument("--no-lstm", action="store_true", help="ใช้ XGB+LGB เท่านั้น (เร็วขึ้น)")
    ap.add_argument("--tail", type=int, default=0, help="ใช้เฉพาะ N แท่งล่าสุด (0=ทั้งหมด)")
    ap.add_argument("--all", action="store_true", help="รันทุก TF")
    args = ap.parse_args()

    tfs = ["M15", "M30", "H1"] if args.all else [args.tf]
    for tf in tfs:
        try:
            run_backtest(tf, conf_min=args.conf, lot=args.lot, spread=args.spread,
                         max_hold=args.max_hold, use_lstm=not args.no_lstm, tail=args.tail)
        except Exception as e:
            print(f"[BT] {tf} error: {e}")
