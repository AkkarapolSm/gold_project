"""
=============================================================
 Gold Dashboard — FastAPI Backend
=============================================================
 ติดตั้ง:
   pip install fastapi uvicorn python-dotenv

 รัน:
   uvicorn gold_server:app --reload --port 8000

 เปิด browser: http://localhost:8000
=============================================================
"""

import json
import threading
from datetime import datetime
from pathlib import Path

import schedule
import time
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

DATA_DIR  = Path("./gold_data")
MODEL_DIR = Path("./gold_models")

app = FastAPI(title="Gold Signal Dashboard")

# serve dashboard HTML
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
def index():
    return FileResponse("dashboard.html")


# ── API: ดึง signals ทุก TF ─────────────────────────────────
@app.get("/api/signals")
def get_signals():
    sig_path = DATA_DIR / "realtime_signals.json"
    log_path = DATA_DIR / "sent_signals"

    result = {}

    # signals จาก ML model
    if sig_path.exists():
        try:
            with open(sig_path, encoding="utf-8") as f:
                data = json.load(f)
            result.update(data.get("signals_by_tf", data))
            if "tick" in data:
                result["tick"] = data["tick"]
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass  # ไฟล์กำลังถูกเขียนอยู่ — ข้ามรอบนี้

    # log วันนี้ — อ่านจาก all_signals ก่อน ถ้าไม่มีค่อย fallback sent_signals
    today = datetime.now().strftime("%Y%m%d")
    log_entries = []
    for tf in ["M15", "M30", "H1"]:
        # อ่าน all_signals ก่อน ถ้าไม่มีค่อย fallback sent_signals
        for log_dir in ["all_signals", "sent_signals"]:
            lp = DATA_DIR / log_dir / f"{tf}_{today}.jsonl"
            if lp.exists():
                with open(lp, encoding="utf-8") as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            sig   = entry.get("signal", {})
                            # แก้ format เวลา — เก็บเป็น HH:MM ตรงๆ
                            raw_time = entry.get("time", "")
                            if "T" in raw_time:
                                # ISO format เช่น 2026-03-23T14:30:00 → 14:30
                                t_part = raw_time.split("T")[-1][:5]
                            else:
                                # เก็บเป็น HH:MM อยู่แล้ว
                                t_part = raw_time[:5]
                            log_entries.append({
                                "time" : t_part,
                                "tf"   : tf,
                                "dir"  : sig.get("direction", ""),
                                "entry": sig.get("entry"),
                                "sl"   : sig.get("SL"),
                                "tp"   : sig.get("TP"),
                                "conf" : sig.get("confidence", ""),
                            })
                        except Exception:
                            pass
                break  # ถ้าเจอ all_signals แล้วไม่ต้อง fallback

    log_entries.sort(key=lambda x: x["time"], reverse=True)
    result["log"] = log_entries[:20]

    # ── สร้าง indicators จาก M15 signal ─────────────────────
    result["indicators"] = _build_indicators(result.get("M15", {}))

    return JSONResponse(result)


def _build_indicators(sig: dict) -> list:
    """แปลงค่าจาก signal dict เป็น indicator list สำหรับ dashboard"""
    if not sig:
        return []

    inds = []

    # RSI
    rsi = sig.get("RSI") or sig.get("RSI14")
    if rsi is not None:
        rsi = float(rsi)
        if rsi < 30:   rsi_sig, rsi_cls = "Oversold",   "up-color"
        elif rsi > 70: rsi_sig, rsi_cls = "Overbought", "down-color"
        else:          rsi_sig, rsi_cls = "Neutral",    "neu-color"
        inds.append({"name":"RSI 14","value":rsi,"min":0,"max":100,
                     "signal":rsi_sig,"sigClass":rsi_cls})

    # ATR
    atr = sig.get("ATR") or sig.get("ATR14")
    if atr is not None:
        inds.append({"name":"ATR 14","value":float(atr),"min":0,"max":30,
                     "signal":"Volatility","sigClass":"gold-color"})

    # Confidence as score
    conf_str = sig.get("confidence","0%")
    try:
        conf_val = float(str(conf_str).replace("%",""))
    except Exception:
        conf_val = 0
    if conf_val > 0:
        if conf_val >= 70:   conf_sig, conf_cls = "Strong",  "up-color"
        elif conf_val >= 60: conf_sig, conf_cls = "Moderate","gold-color"
        else:                conf_sig, conf_cls = "Weak",    "neu-color"
        inds.append({"name":"Confidence","value":conf_val,"min":0,"max":100,
                     "signal":conf_sig,"sigClass":conf_cls})

    # Proba UP/DOWN/NEUTRAL
    proba = sig.get("proba", {})
    if proba:
        try:
            up_val = float(str(proba.get("UP","0%")).replace("%",""))
            inds.append({"name":"Prob UP","value":up_val,"min":0,"max":100,
                         "signal":"UP" if up_val>50 else "—","sigClass":"up-color"})
            dn_val = float(str(proba.get("DOWN","0%")).replace("%",""))
            inds.append({"name":"Prob DOWN","value":dn_val,"min":0,"max":100,
                         "signal":"DOWN" if dn_val>50 else "—","sigClass":"down-color"})
        except Exception:
            pass

    # Score
    score = sig.get("score")
    if score is not None:
        score = float(score)
        inds.append({"name":"Signal Score","value":score+8,"min":0,"max":16,
                     "signal":"Bullish" if score>0 else ("Bearish" if score<0 else "Neutral"),
                     "sigClass":"up-color" if score>0 else ("down-color" if score<0 else "neu-color")})

    return inds


# ── API: tick ปัจจุบัน ───────────────────────────────────────
@app.get("/api/tick")
def get_tick():
    sig_path = DATA_DIR / "realtime_signals.json"
    if sig_path.exists():
        try:
            with open(sig_path, encoding="utf-8") as f:
                data = json.load(f)
            return JSONResponse(data.get("tick", {}))
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
    return JSONResponse({})


# ── Background: รัน pipeline + predict ทุกนาที ────────────────
def _background_loop():
    """รัน pipeline + predict + alert ใน background thread"""
    try:
        import MetaTrader5 as mt5
        from gold_mt5_pipeline import connect_mt5, fetch_candles, TIMEFRAMES
        from gold_features import build_features
        from gold_model import EnsembleModel, SEQ_LEN
        if not connect_mt5():
            print("[BG] MT5 ไม่ได้เชื่อมต่อ — background loop หยุด")
            return

        def job():
            signals_by_tf = {}
            for tf_name, tf_val in TIMEFRAMES.items():
                try:
                    df_raw  = fetch_candles(tf_name, tf_val, n=600)
                    df_feat = build_features(df_raw)
                    ens = EnsembleModel.load_all(tf_name)
                    last_close = float(df_feat["Close"].iloc[-1])
                    atr = float(df_feat["ATR14"].iloc[-1])
                    sig = ens.predict_signal(df_feat, last_close, atr)
                    signals_by_tf[tf_name] = sig
                except Exception as e:
                    print(f"[BG] {tf_name} error: {e}")

            # ดึง tick
            tick = {}
            try:
                from gold_mt5_pipeline import fetch_tick
                tick = fetch_tick()
            except Exception:
                pass

            # บันทึก — ใช้ utf-8 เพื่อรองรับ emoji และ unicode
            out = {"tick": tick, "signals_by_tf": signals_by_tf,
                   "updated": datetime.now().isoformat()}
            with open(DATA_DIR / "realtime_signals.json", "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2, default=str)

            # บันทึก ALL signal log (ทุก signal ไม่ใช่แค่ที่ส่ง LINE)
            today = datetime.now().strftime("%Y%m%d")
            all_log_dir = DATA_DIR / "all_signals"
            all_log_dir.mkdir(exist_ok=True)
            for tf_name, sig in signals_by_tf.items():
                if not sig or sig.get("error"):
                    continue
                log_path = all_log_dir / f"{tf_name}_{today}.jsonl"
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "time"  : datetime.now().strftime("%H:%M"),
                        "tf"    : tf_name,
                        "signal": sig,
                    }, ensure_ascii=False, default=str) + "\n")

            print(f"[BG] ✅ อัปเดต {datetime.now().strftime('%H:%M:%S')}")

        schedule.every(1).minutes.do(job)
        job()   # รันทันทีรอบแรก

        while True:
            schedule.run_pending()
            time.sleep(10)

    except Exception as e:
        print(f"[BG ERROR] {e}")


@app.on_event("startup")
def startup():
    DATA_DIR.mkdir(exist_ok=True)
    t = threading.Thread(target=_background_loop, daemon=True)
    t.start()
    print("🚀 Gold Dashboard เริ่มทำงาน → http://localhost:8000")


if __name__ == "__main__":
    uvicorn.run("gold_server:app", host="0.0.0.0", port=8000, reload=False)