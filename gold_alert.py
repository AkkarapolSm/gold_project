"""
=============================================================
 Gold Alert System — LINE Messaging API
=============================================================
 ติดตั้ง:
   pip install requests schedule python-dotenv

 วิธีสมัคร LINE Messaging API (ฟรี 200 msg/เดือน):
   1. เข้า https://developers.line.biz -> Login
   2. Create Provider -> Create a Messaging API channel
   3. Messaging API -> Issue Channel access token -> Copy
   4. เพิ่ม bot เป็นเพื่อนใน LINE -> ส่งข้อความมา 1 ครั้ง
   5. รัน: python gold_alert.py --get-userid

 สร้างไฟล์ .env:
   LINE_CHANNEL_TOKEN=your_channel_access_token
   LINE_USER_ID=Uxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

 วิธีใช้:
   from gold_alert import AlertManager
   alert = AlertManager()
   alert.send_signal("M15", signal_dict)
=============================================================
"""

import argparse
import json
import os
import schedule
import time
import threading
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import requests

load_dotenv()

DATA_DIR = Path("./gold_data")


# ════════════════════════════════════════════════════════════
#  LINE MESSAGING API
# ════════════════════════════════════════════════════════════

class LineMessaging:
    """
    LINE Messaging API — ส่ง Push Message หา user โดยตรง
    ฟรี 200 messages/เดือน เพียงพอสำหรับ trading alert
    """
    PUSH_URL     = "https://api.line.me/v2/bot/message/push"
    FOLLOWER_URL = "https://api.line.me/v2/bot/followers/ids"

    def __init__(self, channel_token: str = None, user_id: str = None):
        self.channel_token = channel_token or os.getenv("LINE_CHANNEL_TOKEN", "")
        self.user_id       = user_id       or os.getenv("LINE_USER_ID", "")

    def _headers(self) -> dict:
        return {
            "Content-Type" : "application/json",
            "Authorization": f"Bearer {self.channel_token}",
        }

    def send(self, message: str) -> bool:
        """ส่งข้อความ text ธรรมดา (fallback)"""
        if not self.channel_token or not self.user_id:
            print("[LINE] ไม่มี channel_token หรือ user_id — ข้าม")
            return False
        try:
            r = requests.post(self.PUSH_URL, headers=self._headers(), json={
                "to"      : self.user_id,
                "messages": [{"type": "text", "text": message}],
            }, timeout=10)
            ok = r.status_code == 200
            print(f"[LINE] {'OK' if ok else f'Error {r.status_code}: {r.text}'}")
            return ok
        except Exception as e:
            print(f"[LINE] Exception: {e}")
            return False

    def send_flex(self, tf: str, sig: dict) -> bool:
        """ส่ง Flex Message — card สวยงาม มีสีตาม UP/DOWN"""
        if not self.channel_token or not self.user_id:
            return False

        direction = sig.get("direction", "?")
        is_up     = "UP"   in direction
        is_down   = "DOWN" in direction
        color     = "#1DB954" if is_up else ("#E8455A" if is_down else "#888888")
        arrow     = "▲" if is_up else ("▼" if is_down else "—")
        conf      = sig.get("confidence", "?")
        now       = datetime.now().strftime("%d/%m %H:%M")

        def row(label, value, val_color="#333333"):
            return {
                "type": "box", "layout": "horizontal",
                "contents": [
                    {"type": "text", "text": label, "size": "sm",
                     "color": "#888888", "flex": 2},
                    {"type": "text", "text": str(value), "size": "sm",
                     "color": val_color, "align": "end",
                     "flex": 3, "weight": "bold"},
                ]
            }

        proba = sig.get("proba", {})
        flex_body = {
            "type": "bubble", "size": "kilo",
            "header": {
                "type": "box", "layout": "vertical",
                "backgroundColor": color,
                "contents": [
                    {"type": "text",
                     "text": f"{arrow} XAUUSD [{tf}]  {now}",
                     "color": "#FFFFFF", "size": "md", "weight": "bold"},
                    {"type": "text",
                     "text": f"{direction}  .  {conf}",
                     "color": "#FFFFFFCC", "size": "sm"},
                ]
            },
            "body": {
                "type": "box", "layout": "vertical", "spacing": "sm",
                "contents": [
                    row("Entry",       sig.get("entry", "-"),             "#D4A843"),
                    row("Stop loss",   sig.get("SL",    "-"),             "#E8455A"),
                    row("Take profit", sig.get("TP",    "-"),             "#1DB954"),
                    row("R : R",       f"1 : {sig.get('RR','-')}",       "#555555"),
                    row("Pips est",    f"~{sig.get('pips_target','-')}", "#555555"),
                    {"type": "separator", "margin": "md"},
                    {
                        "type": "box", "layout": "horizontal", "margin": "sm",
                        "contents": [
                            {"type": "text",
                             "text": f"UP {proba.get('UP','?')}",
                             "size": "xs", "color": "#1DB954",
                             "flex": 1, "align": "center"},
                            {"type": "text",
                             "text": f"NEU {proba.get('NEUTRAL','?')}",
                             "size": "xs", "color": "#888888",
                             "flex": 1, "align": "center"},
                            {"type": "text",
                             "text": f"DN {proba.get('DOWN','?')}",
                             "size": "xs", "color": "#E8455A",
                             "flex": 1, "align": "center"},
                        ]
                    }
                ]
            }
        }

        try:
            r = requests.post(self.PUSH_URL, headers=self._headers(), json={
                "to"      : self.user_id,
                "messages": [{
                    "type"    : "flex",
                    "altText" : f"XAUUSD {tf} {direction} {conf}",
                    "contents": flex_body,
                }],
            }, timeout=10)
            ok = r.status_code == 200
            if ok:
                print("[LINE Flex] OK")
            else:
                print(f"[LINE Flex] Error {r.status_code}: {r.text}")
                return self.send(_format_text(tf, sig))
            return ok
        except Exception as e:
            print(f"[LINE Flex] Exception: {e}")
            return self.send(_format_text(tf, sig))

    def get_user_ids(self) -> list:
        """ดึง userId ของคนที่ follow bot"""
        if not self.channel_token:
            print("[LINE] ไม่มี channel_token")
            return []
        try:
            r = requests.get(self.FOLLOWER_URL,
                             headers=self._headers(), timeout=10)
            ids = r.json().get("userIds", [])
            print(f"[LINE] พบ {len(ids)} user:")
            for uid in ids:
                print(f"  -> {uid}")
            print("\nคัดลอก userId ใส่ใน .env: LINE_USER_ID=Uxxxxxx")
            return ids
        except Exception as e:
            print(f"[LINE] get_user_ids error: {e}")
            return []


# ════════════════════════════════════════════════════════════
#  HELPER — format text
# ════════════════════════════════════════════════════════════

def _format_text(tf: str, sig: dict) -> str:
    now   = datetime.now().strftime("%H:%M")
    dir_  = sig.get("direction", "?")
    arrow = "UP" if "UP" in dir_ else ("DOWN" if "DOWN" in dir_ else "->")
    lines = [
        f"XAUUSD Signal [{tf}] {now}",
        f"{arrow}  {dir_}  Confidence: {sig.get('confidence','?')}",
        "-------------------",
        f"Entry : {sig.get('entry','-')}",
        f"SL    : {sig.get('SL','-')}",
        f"TP    : {sig.get('TP','-')}",
        f"R:R   : 1:{sig.get('RR','-')}",
        f"Pips  : ~{sig.get('pips_target','-')}",
    ]
    if "proba" in sig:
        p = sig["proba"]
        lines += [
            "-------------------",
            f"UP {p.get('UP','?')}  NEU {p.get('NEUTRAL','?')}  DN {p.get('DOWN','?')}",
        ]
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════
#  ALERT MANAGER
# ════════════════════════════════════════════════════════════

class AlertManager:
    MIN_CONFIDENCE = 60.0
    SIGNAL_DIR     = DATA_DIR / "sent_signals"

    def __init__(self):
        self.line = LineMessaging()
        self.SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
        self._last_sent = {}

    def _is_duplicate(self, tf: str, sig: dict) -> bool:
        key = f"{sig.get('direction','')}_{sig.get('entry','')}"
        if self._last_sent.get(tf) == key:
            return True
        self._last_sent[tf] = key
        return False

    def send_signal(self, tf: str, sig: dict) -> bool:
        if not sig or sig.get("error"):
            return False
        if "NEUTRAL" in sig.get("direction", ""):
            return False

        try:
            conf = float(sig.get("confidence", "0").replace("%", ""))
        except Exception:
            conf = 0
        if conf < self.MIN_CONFIDENCE:
            print(f"[ALERT] {tf} confidence {conf:.1f}% < {self.MIN_CONFIDENCE}% -- skip")
            return False

        if self._is_duplicate(tf, sig):
            print(f"[ALERT] {tf} duplicate -- skip")
            return False

        ok = self.line.send_flex(tf, sig)

        log_path = self.SIGNAL_DIR / f"{tf}_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "time"  : datetime.now().isoformat(),
                "tf"    : tf,
                "signal": sig,
            }, ensure_ascii=False, default=str) + "\n")

        return ok

    def send_all(self, signals: dict):
        for tf, sig in signals.items():
            self.send_signal(tf, sig)

    def start_loop(self, interval_minutes: int = 15):
        """รัน background thread ส่ง alert ทุก N นาที"""
        def _job():
            sig_path = DATA_DIR / "realtime_signals.json"
            if not sig_path.exists():
                return
            try:
                with open(sig_path, encoding="utf-8") as f:
                    data = json.load(f)
                self.send_all(data.get("signals_by_tf", data))
            except Exception as e:
                print(f"[ALERT LOOP] error: {e}")

        schedule.every(interval_minutes).minutes.do(_job)

        def _run():
            while True:
                schedule.run_pending()
                time.sleep(10)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        print(f"[ALERT] loop started every {interval_minutes} min")
        return t


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--get-userid", action="store_true",
                        help="ดึง LINE userId ของ followers")
    args = parser.parse_args()

    if args.get_userid:
        LineMessaging().get_user_ids()
    else:
        alert = AlertManager()
        test_sig = {
            "direction"  : "UP",
            "confidence" : "72.5%",
            "entry"      : 3085.50,
            "SL"         : 3081.80,
            "TP"         : 3095.20,
            "RR"         : 1.67,
            "pips_target": 9.7,
            "proba"      : {"UP": "72.5%", "NEUTRAL": "14.3%", "DOWN": "13.2%"},
        }
        print("-- Test Alert --")
        alert.send_signal("M15", test_sig)