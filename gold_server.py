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
import os
import threading
from datetime import datetime, timedelta
from pathlib import Path

import schedule
import time
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

DATA_DIR  = Path("./gold_data")
MODEL_DIR = Path("./gold_models")

# magic number ของบอทเทรด (ต้องตรงกับ gold_trader.py เพื่อ filter ออเดอร์ของบอท)
MAGIC = int(os.getenv("TRADE_MAGIC", "20250323"))

app = FastAPI(title="Gold Signal Dashboard")

# serve dashboard HTML
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
def index():
    return FileResponse("dashboard.html")


@app.get("/trades")
def trades_page():
    return FileResponse("trades.html")


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


# ── สถิติผลการเทรดจริง (realized) + equity curve จาก MT5 history ──
def _realized_stats(days: int = 30) -> dict:
    """
    คำนวณ win-rate / profit factor / equity curve จาก history deals ของบอท
    (filter ด้วย magic) — ใช้ MT5 ที่ background loop เชื่อมต่อไว้แล้ว
    """
    empty = {"trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
             "realized_pl": 0.0, "profit_factor": 0.0,
             "best": 0.0, "worst": 0.0, "equity_curve": []}
    try:
        import MetaTrader5 as mt5
        to_dt   = datetime.now()
        from_dt = to_dt - timedelta(days=days)
        deals   = mt5.history_deals_get(from_dt, to_dt)
        if not deals:
            return empty

        mine = [d for d in deals if d.magic == MAGIC]
        if not mine:
            return empty

        def net(d):
            return d.profit + d.swap + d.commission

        # รวม P/L + เวลาปิดล่าสุด ต่อ position (เฉพาะขา OUT)
        pos_pnl, pos_time = {}, {}
        for d in mine:
            if d.entry == mt5.DEAL_ENTRY_OUT:
                pos_pnl[d.position_id]  = pos_pnl.get(d.position_id, 0.0) + net(d)
                pos_time[d.position_id] = max(pos_time.get(d.position_id, 0), d.time)

        if not pos_pnl:
            return empty

        # equity curve: เรียงตามเวลาปิด → cumulative realized P/L
        ordered = sorted(pos_pnl.keys(), key=lambda pid: pos_time[pid])
        cum, curve = 0.0, []
        for pid in ordered:
            cum += pos_pnl[pid]
            curve.append({
                "t"  : datetime.fromtimestamp(pos_time[pid]).strftime("%d/%m %H:%M"),
                "pnl": round(cum, 2),
            })

        pnls   = list(pos_pnl.values())
        wins   = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        n      = len(pnls)
        gross_win  = sum(wins)
        gross_loss = -sum(losses)
        pf = round(gross_win / gross_loss, 2) if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)

        return {
            "trades"       : n,
            "wins"         : len(wins),
            "losses"       : len(losses),
            "win_rate"     : round(len(wins) / n * 100, 1) if n else 0.0,
            "realized_pl"  : round(sum(net(d) for d in mine), 2),
            "profit_factor": pf,
            "best"         : round(max(pnls), 2),
            "worst"        : round(min(pnls), 2),
            "equity_curve" : curve[-200:],
        }
    except Exception as e:
        print(f"[API] realized_stats error: {e}")
        return empty


# ── API: ประวัติการเข้าออเดอร์ของบอท + ออเดอร์ที่เปิดอยู่ ──────
@app.get("/api/trades")
def get_trades():
    """อ่าน trade log ของบอท (trades_*.jsonl) + ออเดอร์ที่เปิดอยู่จาก MT5"""
    log_dir = DATA_DIR / "trade_logs"
    trades = []
    if log_dir.exists():
        for fp in sorted(log_dir.glob("trades_*.jsonl")):
            with open(fp, encoding="utf-8") as f:
                for line in f:
                    try:
                        trades.append(json.loads(line))
                    except Exception:
                        pass

    trades.sort(key=lambda x: x.get("time", ""), reverse=True)

    # นับสรุปจาก log
    n_open  = sum(1 for t in trades if t.get("action") == "OPEN")
    n_fail  = sum(1 for t in trades if t.get("action") == "OPEN_FAIL")
    n_close = sum(1 for t in trades if t.get("action") == "CLOSE")

    # ออเดอร์ที่เปิดอยู่ + floating P/L (background loop เขียนไว้ใน realtime_signals.json)
    bot_orders, bot_profit = [], 0.0
    sig_path = DATA_DIR / "realtime_signals.json"
    if sig_path.exists():
        try:
            with open(sig_path, encoding="utf-8") as f:
                d = json.load(f)
            bot_orders = d.get("bot_orders", [])
            bot_profit = d.get("bot_profit", 0.0)
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

    # สถิติผลจริง (realized) + equity curve จาก MT5 history
    stats = _realized_stats(days=30)

    return JSONResponse({
        "trades": trades[:200],
        "orders": bot_orders,
        "summary": {
            "open"       : n_open,
            "fail"       : n_fail,
            "close"      : n_close,
            "open_now"   : len(bot_orders),
            "floating_pl": bot_profit,
        },
        "stats": stats,
    })


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

        # Telegram notifier (signal channel) — no-op ถ้ายังไม่ตั้งค่าใน .env
        try:
            from gold_telegram import TelegramNotifier
            tg = TelegramNotifier()
            if tg.enabled():
                print("[BG] Telegram signal alert: เปิดใช้งาน")
        except Exception as e:
            tg = None
            print(f"[BG] Telegram init error: {e}")

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

            # ส่งสัญญาณเข้า Telegram (ช่อง signal) — มี dedup + กรอง conf ในตัว
            if tg is not None:
                try:
                    tg.send_signals(signals_by_tf)
                except Exception as e:
                    print(f"[BG] Telegram signal error: {e}")

            # ดึง tick
            tick = {}
            try:
                from gold_mt5_pipeline import fetch_tick
                tick = fetch_tick()
            except Exception:
                pass

            # ดึงออเดอร์ที่เปิดอยู่ของบอท (filter ด้วย magic number)
            bot_orders, bot_profit = [], 0.0
            try:
                positions = mt5.positions_get()
                for p in (positions or []):
                    if p.magic != MAGIC:
                        continue
                    typ = "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL"
                    bot_orders.append({
                        "ticket"    : p.ticket,
                        "type"      : typ,
                        "comment"   : p.comment,
                        "volume"    : p.volume,
                        "price_open": round(p.price_open, 2),
                        "sl"        : round(p.sl, 2),
                        "tp"        : round(p.tp, 2),
                        "profit"    : round(p.profit, 2),
                        "time"      : datetime.fromtimestamp(p.time).strftime("%H:%M"),
                    })
                    bot_profit += p.profit
            except Exception as e:
                print(f"[BG] orders error: {e}")

            # บันทึก — ใช้ utf-8 เพื่อรองรับ emoji และ unicode
            out = {"tick": tick, "signals_by_tf": signals_by_tf,
                   "bot_orders": bot_orders, "bot_profit": round(bot_profit, 2),
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