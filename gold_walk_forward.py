"""
=============================================================
 Walk-Forward Optimization
 Train โมเดลใหม่ทุกสัปดาห์ด้วยข้อมูลล่าสุด
=============================================================
 แนวคิด:
   ├── Window size  = 6 เดือน (train)
   ├── Test period  = 2 สัปดาห์ (validate)
   ├── Step size    = 1 สัปดาห์ (เลื่อนทีละสัปดาห์)
   └── รัน auto ทุกวันจันทร์ตี 2 (ตลาดปิด)

 รัน manual:
   python gold_walk_forward.py           # รัน WFO ย้อนหลัง
   python gold_walk_forward.py --retrain # retrain ด้วยข้อมูลใหม่ล่าสุด
=============================================================
"""

import argparse
import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

MODEL_DIR = Path("./gold_models")
DATA_DIR  = Path("./gold_data")
WFO_DIR   = Path("./gold_wfo")
WFO_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

TRAIN_WEEKS = 24   # ใช้ข้อมูลย้อนหลัง 24 สัปดาห์ (≈6 เดือน)
TEST_WEEKS  = 2    # ทดสอบ 2 สัปดาห์
STEP_WEEKS  = 1    # เลื่อนทีละสัปดาห์


# ════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════

def load_feature_df(tf: str) -> pd.DataFrame:
    path = DATA_DIR / f"XAUUSD_{tf}_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"ไม่พบ {path}")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df["label"] = df["target"].map({-1: 0, 0: 1, 1: 2})
    return df.dropna()

def get_feature_cols(df: pd.DataFrame) -> list:
    exclude = {"target","label","target_binary","future_ret_5",
               "future_high","future_low","Open","High","Low",
               "Close","Volume"}
    return [c for c in df.columns if c not in exclude]

def append_fundamental_features(df: pd.DataFrame) -> pd.DataFrame:
    """เพิ่ม fundamental score และ sentiment เป็น feature"""
    fund_path = DATA_DIR / "fundamentals.json"
    sent_path = DATA_DIR / "sentiment.json"

    if fund_path.exists():
        with open(fund_path, encoding="utf-8") as f:
            fund = json.load(f)
        score = fund.get("score", {})
        df["fund_score"]   = score.get("normalized", 0)
        df["dxy_trend"]    = 1 if fund.get("dxy",{}).get("gold_bias") == "bullish" else -1
        df["yield_trend"]  = 1 if fund.get("us10y",{}).get("gold_bias") == "bullish" else -1
        df["fomc_near"]    = 1 if fund.get("fomc",{}).get("high_volatility") else 0

    if sent_path.exists():
        with open(sent_path, encoding="utf-8") as f:
            sent_data = json.load(f)
        sent = sent_data.get("sentiment", {})
        df["sent_score"]   = sent.get("avg_net_score", 0)
        df["sent_bull_pct"]= sent.get("bull_pct", 50)

    return df


# ════════════════════════════════════════════════════════════
#  SINGLE WINDOW TRAINER
# ════════════════════════════════════════════════════════════

def train_window(df_train: pd.DataFrame, df_test: pd.DataFrame,
                 feat_cols: list, model_type: str = "xgb") -> dict:
    """
    Train โมเดลใน window เดียว คืน metrics + model
    """
    import xgboost as xgb
    import lightgbm as lgb

    scaler  = RobustScaler()
    X_train = scaler.fit_transform(df_train[feat_cols])
    y_train = df_train["label"].values
    X_test  = scaler.transform(df_test[feat_cols])
    y_test  = df_test["label"].values

    if model_type == "xgb":
        model = xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=42, n_jobs=-1, verbosity=0,
        )
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)], verbose=False)

    elif model_type == "lgb":
        model = lgb.LGBMClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            num_leaves=31, random_state=42, n_jobs=-1, verbose=-1,
        )
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  callbacks=[lgb.early_stopping(30, verbose=False),
                             lgb.log_evaluation(period=-1)])

    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # directional accuracy (UP vs DOWN เท่านั้น ไม่นับ neutral)
    mask   = (y_test != 1) & (y_pred != 1)
    dir_acc = accuracy_score(y_test[mask], y_pred[mask]) if mask.sum() > 10 else 0

    return {
        "model"    : model,
        "scaler"   : scaler,
        "accuracy" : round(acc, 4),
        "f1"       : round(f1, 4),
        "dir_acc"  : round(dir_acc, 4),
        "n_train"  : len(df_train),
        "n_test"   : len(df_test),
    }


# ════════════════════════════════════════════════════════════
#  WALK-FORWARD BACKTEST
# ════════════════════════════════════════════════════════════

def run_walk_forward(tf: str = "M15",
                     model_type: str = "xgb",
                     verbose: bool = True) -> pd.DataFrame:
    """
    รัน Walk-Forward Optimization แบบ full backtest
    คืน DataFrame ของ performance แต่ละ window
    """
    df = load_feature_df(tf)
    df = append_fundamental_features(df)
    feat_cols = get_feature_cols(df)

    # candles per week
    cpw = {"M15": 4*24*5, "M30": 2*24*5, "H1": 24*5}
    cpp = cpw.get(tf, 480)

    train_n = TRAIN_WEEKS * cpp
    test_n  = TEST_WEEKS  * cpp
    step_n  = STEP_WEEKS  * cpp

    results = []
    start   = train_n
    total   = len(df)
    window  = 0

    if verbose:
        print(f"\n{'═'*55}")
        print(f"  Walk-Forward: {tf} | model={model_type}")
        print(f"  Data: {total} rows | train={TRAIN_WEEKS}wk test={TEST_WEEKS}wk")
        print(f"{'═'*55}")

    while start + test_n <= total:
        df_train = df.iloc[start - train_n : start]
        df_test  = df.iloc[start : start + test_n]

        try:
            res = train_window(df_train, df_test, feat_cols, model_type)
            window += 1
            entry = {
                "window"      : window,
                "train_start" : str(df_train.index[0])[:10],
                "train_end"   : str(df_train.index[-1])[:10],
                "test_start"  : str(df_test.index[0])[:10],
                "test_end"    : str(df_test.index[-1])[:10],
                "accuracy"    : res["accuracy"],
                "f1"          : res["f1"],
                "dir_acc"     : res["dir_acc"],
                "n_train"     : res["n_train"],
                "n_test"      : res["n_test"],
            }
            results.append(entry)
            if verbose:
                print(f"  W{window:02d} [{entry['test_start']}→{entry['test_end']}] "
                      f"acc={res['accuracy']:.3f} dir={res['dir_acc']:.3f} f1={res['f1']:.3f}")
        except Exception as e:
            print(f"  [ERROR] window {window}: {e}")

        start += step_n

    if not results:
        print("[WFO] ข้อมูลไม่พอ — ต้องการ >= 6 เดือน")
        return pd.DataFrame()

    df_res = pd.DataFrame(results)

    # Summary
    print(f"\n{'─'*55}")
    print(f"  Walk-Forward Summary ({tf} {model_type.upper()})")
    print(f"  Windows: {len(df_res)}")
    print(f"  Avg Accuracy : {df_res['accuracy'].mean():.4f} ± {df_res['accuracy'].std():.4f}")
    print(f"  Avg Dir Acc  : {df_res['dir_acc'].mean():.4f} ± {df_res['dir_acc'].std():.4f}")
    print(f"  Avg F1       : {df_res['f1'].mean():.4f}")
    print(f"  Best window  : W{df_res.loc[df_res['dir_acc'].idxmax(),'window']} "
          f"({df_res['dir_acc'].max():.4f})")
    print(f"  Worst window : W{df_res.loc[df_res['dir_acc'].idxmin(),'window']} "
          f"({df_res['dir_acc'].min():.4f})")

    # บันทึก
    out_path = WFO_DIR / f"wfo_{tf}_{model_type}.csv"
    df_res.to_csv(out_path, index=False)
    print(f"\n  💾 บันทึกที่ {out_path}")

    return df_res


# ════════════════════════════════════════════════════════════
#  WEEKLY RETRAIN (รันทุกจันทร์)
# ════════════════════════════════════════════════════════════

def retrain_latest(tf: str = "M15", model_type: str = "xgb") -> dict:
    """
    Train โมเดลเดี่ยว (xgb/lgb) ด้วยข้อมูล 6 เดือนล่าสุด — ใช้ "ประเมิน" เท่านั้น
    ⚠️ บันทึกที่ gold_wfo/ (ไม่ใช่ gold_models/) เพราะ feature/รูปแบบไฟล์ต่างจาก
       EnsembleModel ที่ deploy อยู่ — การ deploy จริงให้ใช้ gold_retrain.py
    """
    import joblib

    print(f"\n[RETRAIN] {tf} {model_type.upper()} — {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    try:
        df = load_feature_df(tf)
        df = append_fundamental_features(df)
        feat_cols = get_feature_cols(df)

        # ใช้ 6 เดือนล่าสุด
        cpw = {"M15": 4*24*5, "M30": 2*24*5, "H1": 24*5}
        cpp = cpw.get(tf, 480)
        use_n = TRAIN_WEEKS * cpp
        df_use = df.iloc[-use_n:] if len(df) > use_n else df

        # train/val split 80/20
        split = int(len(df_use) * 0.8)
        df_train, df_val = df_use.iloc[:split], df_use.iloc[split:]

        res = train_window(df_train, df_val, feat_cols, model_type)

        # บันทึก model (ที่ gold_wfo/ เพื่อไม่ชนกับ live models ใน gold_models/)
        save_path = WFO_DIR / f"{model_type}_{tf}_wfo.pkl"
        joblib.dump({
            "model"     : res["model"],
            "scaler"    : res["scaler"],
            "feat_cols" : feat_cols,
            "trained_at": datetime.now().isoformat(),
            "accuracy"  : res["accuracy"],
            "dir_acc"   : res["dir_acc"],
        }, save_path)

        # บันทึก performance log
        log_path = WFO_DIR / f"retrain_log_{tf}_{model_type}.jsonl"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "date"      : datetime.now().isoformat(),
                "tf"        : tf, "model": model_type,
                "accuracy"  : res["accuracy"],
                "dir_acc"   : res["dir_acc"],
                "f1"        : res["f1"],
                "n_train"   : res["n_train"],
                "n_test"    : res["n_test"],
            }) + "\n")

        print(f"  ✅ acc={res['accuracy']:.4f} dir={res['dir_acc']:.4f} → บันทึกที่ {save_path}")
        return res

    except Exception as e:
        print(f"  ❌ Retrain error: {e}")
        return {}

def retrain_all():
    """Retrain ทุก TF ทุก model — รันทุกวันจันทร์"""
    print(f"\n{'═'*55}")
    print(f"  Weekly Retrain — {datetime.now().strftime('%Y-%m-%d')}")
    print(f"{'═'*55}")

    results = {}
    for tf in ["M15", "M30", "H1"]:
        for model_type in ["xgb", "lgb"]:
            key = f"{tf}_{model_type}"
            results[key] = retrain_latest(tf, model_type)

    # สรุปผล
    print(f"\n{'─'*55}")
    print(f"  Retrain Summary")
    for key, r in results.items():
        if r:
            print(f"  {key:12s}: acc={r.get('accuracy',0):.3f} dir={r.get('dir_acc',0):.3f}")
    print(f"{'─'*55}")
    return results


# ════════════════════════════════════════════════════════════
#  AUTO SCHEDULER (background thread)
# ════════════════════════════════════════════════════════════

def start_weekly_retrain_scheduler():
    """เรียกจาก gold_server.py เพื่อให้ retrain อัตโนมัติทุกจันทร์ตี 2"""
    import schedule
    import threading
    import time

    def _job():
        try:
            # อัปเดต features ก่อน
            from gold_mt5_pipeline import connect_mt5, fetch_candles, TIMEFRAMES
            from gold_features import build_features

            if connect_mt5():
                import MetaTrader5 as mt5
                for tf_name, tf_val in TIMEFRAMES.items():
                    df_raw = fetch_candles(tf_name, tf_val, n=3000)
                    df_feat = build_features(df_raw, target_pips=3.0)
                    df_feat.to_csv(DATA_DIR / f"XAUUSD_{tf_name}_features.csv")
                mt5.shutdown()

            retrain_all()
        except Exception as e:
            print(f"[SCHEDULER] retrain error: {e}")

    schedule.every().monday.at("02:00").do(_job)
    print("[SCHEDULER] ตั้ง retrain ทุกวันจันทร์ 02:00")

    def _run():
        while True:
            schedule.run_pending()
            import time
            time.sleep(60)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true",
                        help="Retrain ด้วยข้อมูลล่าสุด (ไม่รัน full backtest)")
    parser.add_argument("--tf", default="M15", choices=["M15","M30","H1"])
    parser.add_argument("--model", default="xgb", choices=["xgb","lgb"])
    args = parser.parse_args()

    if args.retrain:
        retrain_latest(args.tf, args.model)
    else:
        run_walk_forward(args.tf, args.model)