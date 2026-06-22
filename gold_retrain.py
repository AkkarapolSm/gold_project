"""
=============================================================
 Gold Retrain Orchestrator (D1) — เทรนใหม่อัตโนมัติ + gate ก่อน deploy
=============================================================
 ขั้นตอน (ปลอดภัย ไม่ทำลายโมเดลที่ใช้งานอยู่):
   1. (option) rebuild features จาก MT5 ด้วยข้อมูลล่าสุด
   2. ประเมินด้วย Walk-Forward (out-of-sample) → avg directional accuracy
   3. ผ่าน gate (>= RETRAIN_GATE_DIR_ACC) เท่านั้นจึง deploy
   4. deploy = เทรน EnsembleModel ใหม่ (รูปแบบไฟล์เดียวกับที่ใช้งานจริง)
      โดย backup โมเดลเดิมก่อน + rollback อัตโนมัติถ้า error
   5. log ผล + (option) แจ้ง Telegram

 ⚠️ ต่างจาก gold_walk_forward.retrain_latest ซึ่ง train โมเดลเดี่ยวเพื่อ
    "ประเมิน" และเซฟไว้ที่ gold_wfo/ — ตัวนี้คือ path สำหรับ "deploy จริง"

 รัน:
   python gold_retrain.py                 # rebuild + evaluate + deploy (ผ่าน gate)
   python gold_retrain.py --dry-run       # ประเมินอย่างเดียว ไม่ deploy
   python gold_retrain.py --no-rebuild    # ใช้ feature CSV เดิม (ไม่แตะ MT5)
   python gold_retrain.py --tf M15 --gate 0.52
=============================================================
"""

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

MODEL_DIR  = Path("./gold_models")
DATA_DIR   = Path("./gold_data")
WFO_DIR    = Path("./gold_wfo")
BACKUP_DIR = MODEL_DIR / "backup"
WFO_DIR.mkdir(exist_ok=True)

GATE_DIR_ACC = float(os.getenv("RETRAIN_GATE_DIR_ACC", "0.50"))   # OOS dir_acc ขั้นต่ำจึง deploy
TFS = ["M15", "M30", "H1"]


# ── 1. rebuild features ──────────────────────────────────
def rebuild_features() -> bool:
    try:
        import MetaTrader5 as mt5
        from gold_mt5_pipeline import connect_mt5, fetch_candles
        from gold_features import build_features
    except Exception as e:
        print(f"[RETRAIN] import MT5/pipeline ไม่ได้ — ข้าม rebuild ({e})")
        return False

    if not connect_mt5():
        print("[RETRAIN] MT5 เชื่อมต่อไม่ได้ — ใช้ feature CSV เดิม")
        return False

    tf_map = [("M15", mt5.TIMEFRAME_M15), ("M30", mt5.TIMEFRAME_M30), ("H1", mt5.TIMEFRAME_H1)]
    for tf_name, tf_val in tf_map:
        try:
            df = fetch_candles(tf_name, tf_val, n=10000)
            if df is None or df.empty:
                print(f"  [SKIP] {tf_name} ดึงข้อมูลไม่ได้")
                continue
            feat = build_features(df, target_pips=3.0)
            feat.to_csv(DATA_DIR / f"XAUUSD_{tf_name}_features.csv")
            print(f"  ✅ features {tf_name}: {len(feat)} rows")
        except Exception as e:
            print(f"  [ERROR] features {tf_name}: {e}")
    mt5.shutdown()
    return True


# ── 2. evaluate (walk-forward, out-of-sample) ────────────
def evaluate(tf: str, model_type: str = "xgb") -> float | None:
    from gold_walk_forward import run_walk_forward
    df_res = run_walk_forward(tf, model_type, verbose=False)
    if df_res is None or df_res.empty:
        return None
    return round(float(df_res["dir_acc"].mean()), 4)


# ── 3-4. backup / rollback / deploy ──────────────────────
def backup_models() -> Path | None:
    files = list(MODEL_DIR.glob("*.pkl")) + list(MODEL_DIR.glob("meta_*.json"))
    if not files:
        return None
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    dest = BACKUP_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    dest.mkdir(exist_ok=True)
    for p in files:
        shutil.copy2(p, dest / p.name)
    print(f"[RETRAIN] backup โมเดลเดิม → {dest}")
    return dest


def restore_models(backup: Path):
    if not backup or not backup.exists():
        return
    for p in backup.glob("*"):
        shutil.copy2(p, MODEL_DIR / p.name)
    print(f"[RETRAIN] rollback จาก {backup}")


def deploy_retrain(tf: str):
    """เทรน EnsembleModel ใหม่ + เซฟทับ (รูปแบบไฟล์เดียวกับที่ใช้งานจริง)"""
    from gold_model import load_data, EnsembleModel
    df, feat_cols, _ = load_data(tf)
    ens = EnsembleModel()
    ens.fit(df, feat_cols, tf)   # _save_all() เขียน xgb_/lgb_/lstm_/meta_ ให้เอง
    return ens


# ── orchestrator ─────────────────────────────────────────
def run(rebuild: bool = True, gate: float = None,
        deploy: bool = True, tfs: list = None, notify: bool = False) -> dict:
    gate = GATE_DIR_ACC if gate is None else gate
    tfs  = tfs or TFS
    log  = {"date": datetime.now().isoformat(), "gate": gate,
            "deploy": deploy, "results": {}}

    print(f"\n{'═'*56}")
    print(f"  RETRAIN — {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
          f"gate dir_acc>={gate} | deploy={deploy}")
    print(f"{'═'*56}")

    if rebuild:
        rebuild_features()

    backup = backup_models() if deploy else None
    deployed, skipped = [], []

    try:
        for tf in tfs:
            if not (DATA_DIR / f"XAUUSD_{tf}_features.csv").exists():
                print(f"[RETRAIN] {tf}: ไม่มี feature CSV — ข้าม")
                log["results"][tf] = {"note": "no features"}
                continue

            dir_acc = evaluate(tf, "xgb")
            entry = {"dir_acc": dir_acc, "deployed": False}
            print(f"[RETRAIN] {tf}: WFO avg dir_acc = {dir_acc}")

            if dir_acc is None:
                entry["note"] = "WFO ข้อมูลไม่พอ"
            elif dir_acc < gate:
                entry["note"] = f"ไม่ผ่าน gate ({gate}) — คงโมเดลเดิม"
                skipped.append(tf)
            elif deploy:
                deploy_retrain(tf)
                entry["deployed"] = True
                deployed.append(tf)
            else:
                entry["note"] = "ผ่าน gate (dry-run — ไม่ deploy)"
            log["results"][tf] = entry

    except Exception as e:
        print(f"[RETRAIN] ❌ ERROR: {e} — กำลัง rollback")
        restore_models(backup)
        log["error"] = str(e)

    log["deployed"], log["skipped"] = deployed, skipped

    with open(WFO_DIR / "retrain_runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log, ensure_ascii=False) + "\n")

    print(f"\n[RETRAIN] เสร็จ | deploy: {deployed or '-'} | ข้าม: {skipped or '-'}")

    if notify:
        _notify(log)
    return log


def _notify(log: dict):
    try:
        from gold_telegram import TelegramNotifier
        tg = TelegramNotifier()
        if not tg.enabled():
            return
        lines = [f"🔄 Retrain {datetime.now().strftime('%d/%m %H:%M')}  (gate {log['gate']})"]
        for tf, r in log.get("results", {}).items():
            mark = "✅ deploy" if r.get("deployed") else "⏸ คงเดิม"
            lines.append(f"{tf}: dir_acc={r.get('dir_acc')} {mark}")
        if log.get("error"):
            lines.append(f"⚠️ error: {log['error']} (rollback แล้ว)")
        tg._send(tg.order_chat, "\n".join(lines))
    except Exception as e:
        print(f"[RETRAIN] telegram error: {e}")


def start_retrain_scheduler():
    """เรียกจาก gold_server (opt-in ผ่าน env RETRAIN_AUTO) — รันทุกจันทร์ 03:00"""
    import schedule, threading, time

    day  = os.getenv("RETRAIN_DAY",  "monday").strip().lower()
    at_t = os.getenv("RETRAIN_TIME", "03:00").strip()

    def _job():
        try:
            run(rebuild=True, deploy=True, notify=True)
        except Exception as e:
            print(f"[RETRAIN-SCHED] {e}")

    slot = getattr(schedule.every(), day, None)
    if slot is None:
        print(f"[RETRAIN-SCHED] วันไม่ถูกต้อง: {day}")
        return None
    slot.at(at_t).do(_job)
    print(f"[RETRAIN-SCHED] ตั้ง retrain อัตโนมัติทุก {day} {at_t}")

    def _run():
        while True:
            schedule.run_pending()
            time.sleep(60)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Gold retrain orchestrator")
    ap.add_argument("--dry-run", action="store_true", help="ประเมินอย่างเดียว ไม่ deploy")
    ap.add_argument("--no-rebuild", action="store_true", help="ไม่ rebuild features (ใช้ CSV เดิม)")
    ap.add_argument("--tf", choices=TFS, help="จำกัดเฉพาะ TF เดียว")
    ap.add_argument("--gate", type=float, help="override gate dir_acc")
    ap.add_argument("--notify", action="store_true", help="แจ้งผลเข้า Telegram")
    args = ap.parse_args()

    run(rebuild=not args.no_rebuild,
        gate=args.gate,
        deploy=not args.dry_run,
        tfs=[args.tf] if args.tf else None,
        notify=args.notify)
