"""
=============================================================
 Gold Telegram Notifier — ส่ง Signal + Order แยกคนละช่อง
=============================================================
 ติดตั้ง:
   pip install requests python-dotenv   (มีอยู่แล้วใน requirements.txt)

 วิธีตั้งค่า:
   1. คุยกับ @BotFather ใน Telegram -> /newbot -> ได้ Bot Token
   2. สร้าง 2 ช่อง (group/channel) แล้วเพิ่ม bot เข้าไปเป็น admin
        - ช่อง Signal  (ส่งสัญญาณทำนาย)
        - ช่อง Order   (ส่งการเข้า/ปิดออเดอร์ของบอท)
   3. ส่งข้อความอะไรก็ได้ในแต่ละช่อง 1 ครั้ง แล้วรัน:
        python gold_telegram.py --get-chatid
      จะเห็น chat_id ของแต่ละช่อง (ช่อง/กลุ่มจะเป็นเลขติดลบ เช่น -1001234567890)
   4. ใส่ใน .env:
        TELEGRAM_BOT_TOKEN=123456:ABC...
        TELEGRAM_SIGNAL_CHAT_ID=-100xxxxxxxxxx
        TELEGRAM_ORDER_CHAT_ID=-100yyyyyyyyyy
   5. ทดสอบส่ง:
        python gold_telegram.py --test
=============================================================
"""

import argparse
import os
from datetime import datetime

import requests
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
SIGNAL_CHAT = os.getenv("TELEGRAM_SIGNAL_CHAT_ID", "")
ORDER_CHAT  = os.getenv("TELEGRAM_ORDER_CHAT_ID", "")
# ส่ง signal เฉพาะที่ confidence >= ค่านี้ (กันสแปม)
SIGNAL_MIN_CONF = float(os.getenv("TELEGRAM_SIGNAL_MIN_CONF", "70"))

API_BASE = f"https://api.telegram.org/bot{BOT_TOKEN}"


class TelegramNotifier:
    """ส่งข้อความเข้า Telegram — แยกช่อง signal / order"""

    def __init__(self):
        self.token       = BOT_TOKEN
        self.signal_chat = SIGNAL_CHAT
        self.order_chat  = ORDER_CHAT
        self.min_conf    = SIGNAL_MIN_CONF
        self._last_sent  = {}   # dedup signal ต่อ TF

    def enabled(self) -> bool:
        return bool(self.token)

    # ── ส่งข้อความดิบ ────────────────────────────────────
    def _send(self, chat_id: str, text: str) -> bool:
        if not self.token or not chat_id:
            return False
        try:
            r = requests.post(f"{API_BASE}/sendMessage", json={
                "chat_id"               : chat_id,
                "text"                  : text,
                "disable_web_page_preview": True,
            }, timeout=10)
            ok = r.status_code == 200
            if not ok:
                print(f"[TG] error {r.status_code}: {r.text}")
            return ok
        except Exception as e:
            print(f"[TG] exception: {e}")
            return False

    # ── SIGNAL ───────────────────────────────────────────
    def _is_duplicate(self, tf: str, sig: dict) -> bool:
        key = f"{sig.get('direction','')}_{sig.get('entry','')}"
        if self._last_sent.get(tf) == key:
            return True
        self._last_sent[tf] = key
        return False

    def send_signal(self, tf: str, sig: dict) -> bool:
        """ส่งสัญญาณเข้าช่อง signal (กรอง NEUTRAL / conf / กันส่งซ้ำ)"""
        if not self.enabled() or not self.signal_chat:
            return False
        if not sig or sig.get("error"):
            return False
        direction = sig.get("direction", "")
        if "NEUTRAL" in direction or not direction:
            return False

        try:
            conf = float(str(sig.get("confidence", "0")).replace("%", ""))
        except ValueError:
            conf = 0.0
        if conf < self.min_conf:
            return False
        if self._is_duplicate(tf, sig):
            return False

        emoji = "🟢" if "UP" in direction else "🔴"
        proba = sig.get("proba", {})
        now   = datetime.now().strftime("%d/%m %H:%M")
        lines = [
            f"{emoji} XAUUSD Signal [{tf}]  {now}",
            f"{direction}  |  Conf {sig.get('confidence','?')}",
            "──────────────",
            f"Entry : {sig.get('entry','-')}",
            f"SL    : {sig.get('SL','-')}",
            f"TP    : {sig.get('TP','-')}",
            f"R:R   : 1:{sig.get('RR','-')}",
            f"Pips  : ~{sig.get('pips_target','-')}",
        ]
        if proba:
            lines += ["──────────────",
                      f"UP {proba.get('UP','?')}  ·  DN {proba.get('DOWN','?')}"]
        return self._send(self.signal_chat, "\n".join(lines))

    def send_signals(self, signals: dict):
        for tf, sig in (signals or {}).items():
            self.send_signal(tf, sig)

    # ── ORDER ────────────────────────────────────────────
    def send_order(self, action: str, tf: str, direction: str,
                   price, sl, tp, conf, ticket, lot=0.01, note: str = "") -> bool:
        """ส่งการเข้า/ปิดออเดอร์เข้าช่อง order"""
        if not self.enabled() or not self.order_chat:
            return False

        now = datetime.now().strftime("%d/%m %H:%M:%S")
        if action == "OPEN":
            head = "✅ เปิดออเดอร์"
            side = "BUY" if "UP" in (direction or "") else "SELL"
            lines = [
                f"{head}  [{tf}]  {side}",
                f"Ticket : #{ticket}",
                "──────────────",
                f"Price  : {price}",
                f"SL     : {sl}",
                f"TP     : {tp}",
                f"Conf   : {conf}%   |   Lot {lot}",
                f"เวลา   : {now}",
            ]
        elif action == "CLOSE":
            head = "🏁 ปิดออเดอร์"
            lines = [
                f"{head}  #{ticket}",
                f"Price  : {price}",
                f"{note}",
                f"เวลา   : {now}",
            ]
        else:  # OPEN_FAIL หรืออื่นๆ
            lines = [
                f"⚠️ {action}  [{tf}]  {direction}",
                f"{note}",
                f"เวลา   : {now}",
            ]
        return self._send(self.order_chat, "\n".join(lines))


# ════════════════════════════════════════════════════════════
#  CLI helpers
# ════════════════════════════════════════════════════════════

def get_chat_ids():
    """ดึง chat_id จาก getUpdates (ต้องส่งข้อความในช่องนั้นก่อน 1 ครั้ง)"""
    if not BOT_TOKEN:
        print("[TG] ไม่มี TELEGRAM_BOT_TOKEN ใน .env")
        return
    try:
        r = requests.get(f"{API_BASE}/getUpdates", timeout=10)
        data = r.json()
    except Exception as e:
        print(f"[TG] getUpdates error: {e}")
        return

    if not data.get("ok"):
        print(f"[TG] API error: {data}")
        return

    seen = {}
    for upd in data.get("result", []):
        msg = upd.get("message") or upd.get("channel_post") or {}
        chat = msg.get("chat", {})
        cid  = chat.get("id")
        if cid is not None and cid not in seen:
            seen[cid] = chat.get("title") or chat.get("username") or chat.get("type", "")
    if not seen:
        print("ยังไม่พบ chat — ส่งข้อความในช่อง/กลุ่มที่เพิ่ม bot ไว้ 1 ครั้ง แล้วรันใหม่")
        return
    print("พบ chat ดังนี้ (เอาเลขใส่ .env):")
    for cid, name in seen.items():
        print(f"  chat_id = {cid}   ({name})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gold Telegram Notifier")
    parser.add_argument("--get-chatid", action="store_true",
                        help="ดึง chat_id ของช่องที่เพิ่ม bot ไว้")
    parser.add_argument("--test", action="store_true",
                        help="ส่งข้อความทดสอบเข้าทั้ง 2 ช่อง")
    args = parser.parse_args()

    if args.get_chatid:
        get_chat_ids()
    elif args.test:
        tg = TelegramNotifier()
        if not tg.enabled():
            print("[TG] ยังไม่ได้ตั้ง TELEGRAM_BOT_TOKEN ใน .env")
        else:
            ok1 = tg._send(tg.signal_chat, "🟢 ทดสอบช่อง SIGNAL — เชื่อมต่อ Telegram สำเร็จ")
            ok2 = tg._send(tg.order_chat,  "✅ ทดสอบช่อง ORDER — เชื่อมต่อ Telegram สำเร็จ")
            print(f"signal channel: {'OK' if ok1 else 'FAIL'}")
            print(f"order  channel: {'OK' if ok2 else 'FAIL'}")
    else:
        parser.print_help()
