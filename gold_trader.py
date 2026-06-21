"""
=============================================================
 Gold Auto Trader — MT5 Order Execution Bot
=============================================================
 ติดตั้ง:
   pip install MetaTrader5 pandas python-dotenv

 .env:
   MT5_LOGIN=213596035
   MT5_PASSWORD=your_password
   MT5_SERVER=Exness-MT5Real
   TRADE_LOT=0.01           # lot size ต่อ order (default 0.01)
   TRADE_MIN_CONF=80.0      # confidence ขั้นต่ำ (default 80%)
   TRADE_MAX_ORDERS=5       # max open orders รวมทุก TF (default 5)
   TRADE_MAGIC=20250323     # magic number สำหรับ identify orders

 วิธีใช้:
   python gold_trader.py            # รัน bot loop ทุก 1 นาที
   python gold_trader.py --once     # รันครั้งเดียวแล้วออก
   python gold_trader.py --status   # ดู open orders ปัจจุบัน
   python gold_trader.py --closeall # ปิด orders ทั้งหมดของ bot

 หมายเหตุ:
   - bot จะอ่าน signal จาก gold_data/realtime_signals.json
     ที่ถูก generate โดย gold_server.py (background loop)
   - Pyramid: เปิด order ซ้อนได้ ถ้า signal ใหม่ conf >= MIN_CONF
   - TP/SL: ใช้ค่าจาก signal (ATR-based) โดยตรง
   - MAX_ORDERS: safety limit กันเปิดมากเกินไป
=============================================================
"""

import argparse
import json
import os
import time
import schedule
import logging
from datetime import datetime
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ─── Config ───────────────────────────────────────────────
SYMBOL        = "XAUUSDm"          # จะถูก auto-detect ตอน connect
DATA_DIR      = Path("./gold_data")
LOG_DIR       = Path("./gold_data/trade_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── ดึงค่า config จาก .env (มี default ปลอดภัย) ─────────
LOT_SIZE      = float(os.getenv("TRADE_LOT",         "0.01"))
MIN_CONF      = float(os.getenv("TRADE_MIN_CONF",    "80.0"))
MAX_ORDERS    = int(  os.getenv("TRADE_MAX_ORDERS",  "5"))
MAGIC         = int(  os.getenv("TRADE_MAGIC",       "20250323"))

MT5_LOGIN     = int(os.getenv("MT5_LOGIN",    "0"))
MT5_PASSWORD  = os.getenv("MT5_PASSWORD", "")
MT5_SERVER    = os.getenv("MT5_SERVER",   "Exness-MT5Real")

# ── Timeframes ที่ bot จะดู signal ──────────────────────
TIMEFRAMES    = ["M15", "M30", "H1"]

# ── Logging ──────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(message)s",
    datefmt = "%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            LOG_DIR / f"trader_{datetime.now().strftime('%Y%m%d')}.log",
            encoding="utf-8"
        ),
    ]
)
log = logging.getLogger("GoldTrader")

# Telegram notifier (order channel) — no-op ถ้ายังไม่ตั้งค่าใน .env
try:
    from gold_telegram import TelegramNotifier
    tg = TelegramNotifier()
    if tg.enabled():
        log.info("Telegram order alert: เปิดใช้งาน")
except Exception as e:
    tg = None
    log.warning(f"Telegram init error: {e}")


# ════════════════════════════════════════════════════════════
#  MT5 CONNECTION
# ════════════════════════════════════════════════════════════

def connect_mt5() -> bool:
    """เชื่อมต่อ MT5 และ auto-detect symbol"""
    global SYMBOL

    if MT5_LOGIN and MT5_PASSWORD:
        ok = mt5.initialize(
            login    = MT5_LOGIN,
            password = MT5_PASSWORD,
            server   = MT5_SERVER,
        )
    else:
        ok = mt5.initialize()

    if not ok:
        err = mt5.last_error()
        log.error(f"MT5 initialize ล้มเหลว: {err}")
        return False

    # auto-detect symbol
    for sym_try in ["XAUUSDm", "XAUUSD", "GOLD", "XAUUSDc"]:
        sym_info = mt5.symbol_info(sym_try)
        if sym_info is not None:
            SYMBOL = sym_try
            if not sym_info.visible:
                mt5.symbol_select(SYMBOL, True)
            log.info(f"เชื่อมต่อ MT5 สำเร็จ | Symbol: {SYMBOL} | "
                     f"Bid: {sym_info.bid:.2f} Ask: {sym_info.ask:.2f}")
            return True

    log.error("ไม่พบ Symbol XAUUSDm / XAUUSD / GOLD")
    return False


# ════════════════════════════════════════════════════════════
#  SIGNAL READER
# ════════════════════════════════════════════════════════════

def load_signals() -> dict:
    """
    อ่าน realtime_signals.json ที่ gold_server สร้างไว้
    คืน dict: { "M15": {...}, "M30": {...}, "H1": {...} }
    """
    path = DATA_DIR / "realtime_signals.json"
    if not path.exists():
        log.warning("ไม่พบ realtime_signals.json — รอ gold_server สร้างไฟล์ก่อน")
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # รองรับทั้ง { "M15": {...} } และ { "signals_by_tf": { "M15": {...} } }
        if "signals_by_tf" in data:
            return data["signals_by_tf"]
        # กรอง key ที่เป็น TF จริงๆ
        return {k: v for k, v in data.items() if k in TIMEFRAMES}
    except Exception as e:
        log.error(f"อ่าน signal error: {e}")
        return {}


def parse_signal(tf: str, sig: dict) -> dict | None:
    """
    ตรวจสอบ signal ว่าพร้อม execute ไหม
    คืน None ถ้าไม่ผ่านเงื่อนไข
    """
    if not sig or sig.get("error"):
        return None

    direction = sig.get("direction", "")
    if "NEUTRAL" in direction or not direction:
        return None

    # parse confidence
    conf_str = str(sig.get("confidence", "0")).replace("%", "")
    try:
        conf = float(conf_str)
    except ValueError:
        return None

    if conf < MIN_CONF:
        log.debug(f"[{tf}] conf {conf:.1f}% < {MIN_CONF}% — ข้าม")
        return None

    entry = sig.get("entry")
    sl    = sig.get("SL")
    tp    = sig.get("TP")

    if entry is None or sl is None or tp is None:
        log.warning(f"[{tf}] signal ไม่มี entry/SL/TP — ข้าม")
        return None

    return {
        "tf"        : tf,
        "direction" : direction,
        "conf"      : conf,
        "entry"     : float(entry),
        "sl"        : float(sl),
        "tp"        : float(tp),
        "rr"        : sig.get("RR", "-"),
    }


# ════════════════════════════════════════════════════════════
#  ORDER MANAGEMENT
# ════════════════════════════════════════════════════════════

def get_open_orders() -> list:
    """
    ดึง open positions ทั้งหมดของ bot (filter ด้วย magic number)
    """
    positions = mt5.positions_get(symbol=SYMBOL)
    if positions is None:
        return []
    return [p for p in positions if p.magic == MAGIC]


def count_open_by_tf(open_orders: list, tf: str) -> dict:
    """
    นับ open orders แยก BUY/SELL ของ TF นั้น
    โดยอ่านชื่อ comment ที่เราตั้งไว้ตอนเปิด order
    """
    buy_count  = sum(1 for p in open_orders
                     if p.comment.startswith(f"Gold_{tf}") and p.type == mt5.ORDER_TYPE_BUY)
    sell_count = sum(1 for p in open_orders
                     if p.comment.startswith(f"Gold_{tf}") and p.type == mt5.ORDER_TYPE_SELL)
    return {"buy": buy_count, "sell": sell_count}


def get_current_price() -> tuple[float, float]:
    """คืน (bid, ask) ราคาปัจจุบัน"""
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        return 0.0, 0.0
    return tick.bid, tick.ask


def normalize_price(price: float) -> float:
    """ปัดราคาให้ตรงกับ digits ของ symbol"""
    info = mt5.symbol_info(SYMBOL)
    if info is None:
        return round(price, 2)
    digits = info.digits
    return round(price, digits)


# ── เปิด Order ───────────────────────────────────────────

def open_order(parsed: dict) -> bool:
    """
    ส่ง market order ไปที่ MT5
    parsed = { tf, direction, conf, entry, sl, tp }
    """
    tf        = parsed["tf"]
    direction = parsed["direction"]
    sl        = normalize_price(parsed["sl"])
    tp        = normalize_price(parsed["tp"])
    bid, ask  = get_current_price()

    if bid == 0:
        log.error("ไม่สามารถดึงราคาได้ — ข้าม")
        return False

    if "UP" in direction:
        order_type = mt5.ORDER_TYPE_BUY
        price      = ask          # BUY ใช้ ask
    else:
        order_type = mt5.ORDER_TYPE_SELL
        price      = bid          # SELL ใช้ bid

    comment = f"Gold_{tf}_{direction[:2]}_{int(parsed['conf'])}"

    request = {
        "action"      : mt5.TRADE_ACTION_DEAL,
        "symbol"      : SYMBOL,
        "volume"      : LOT_SIZE,
        "type"        : order_type,
        "price"       : price,
        "sl"          : sl,
        "tp"          : tp,
        "deviation"   : 20,           # max slippage 2 pips
        "magic"       : MAGIC,
        "comment"     : comment,
        "type_time"   : mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)

    if result is None:
        log.error(f"[{tf}] order_send คืน None: {mt5.last_error()}")
        return False

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        log.info(
            f"✅ เปิด order สำเร็จ | {tf} | {direction} | "
            f"Lot:{LOT_SIZE} | Price:{price:.2f} | "
            f"SL:{sl:.2f} | TP:{tp:.2f} | "
            f"Conf:{parsed['conf']:.1f}% | Ticket:#{result.order}"
        )
        _log_trade("OPEN", tf, direction, price, sl, tp,
                   parsed["conf"], result.order)
        if tg:
            try:
                tg.send_order("OPEN", tf, direction, round(price, 2),
                              sl, tp, round(parsed["conf"], 1),
                              result.order, lot=LOT_SIZE)
            except Exception as e:
                log.warning(f"Telegram order error: {e}")
        return True
    else:
        log.error(
            f"❌ เปิด order ล้มเหลว | {tf} | retcode={result.retcode} | "
            f"comment={result.comment}"
        )
        _log_trade("OPEN_FAIL", tf, direction, price, sl, tp,
                   parsed["conf"], 0, note=f"retcode={result.retcode}")
        return False


# ── ปิด Order (manual close) ────────────────────────────

def close_order(position) -> bool:
    """ปิด position ด้วย market order"""
    bid, ask = get_current_price()

    if position.type == mt5.ORDER_TYPE_BUY:
        close_type = mt5.ORDER_TYPE_SELL
        price      = bid
    else:
        close_type = mt5.ORDER_TYPE_BUY
        price      = ask

    request = {
        "action"      : mt5.TRADE_ACTION_DEAL,
        "symbol"      : SYMBOL,
        "volume"      : position.volume,
        "type"        : close_type,
        "position"    : position.ticket,
        "price"       : price,
        "deviation"   : 20,
        "magic"       : MAGIC,
        "comment"     : f"close_{position.ticket}",
        "type_time"   : mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        profit = position.profit
        log.info(
            f"✅ ปิด order #{position.ticket} | "
            f"Profit: {profit:+.2f} USD"
        )
        _log_trade("CLOSE", position.comment.split("_")[1] if "_" in position.comment else "?",
                   "CLOSE", price, 0, 0, 0, position.ticket,
                   note=f"profit={profit:+.2f}")
        if tg:
            try:
                tg.send_order("CLOSE", "-", "CLOSE", round(price, 2),
                              0, 0, 0, position.ticket,
                              note=f"Profit: {profit:+.2f} USD")
            except Exception as e:
                log.warning(f"Telegram order error: {e}")
        return True
    else:
        retcode = result.retcode if result else "None"
        log.error(f"❌ ปิด order #{position.ticket} ล้มเหลว: retcode={retcode}")
        return False


def close_all_bot_orders():
    """ปิด orders ทั้งหมดของ bot"""
    orders = get_open_orders()
    if not orders:
        log.info("ไม่มี open orders")
        return
    log.info(f"กำลังปิด {len(orders)} orders...")
    for pos in orders:
        close_order(pos)


# ════════════════════════════════════════════════════════════
#  TRADE LOGGER
# ════════════════════════════════════════════════════════════

def _log_trade(action: str, tf: str, direction: str,
               price: float, sl: float, tp: float,
               conf: float, ticket: int, note: str = ""):
    """บันทึก trade log เป็น JSONL"""
    today    = datetime.now().strftime("%Y%m%d")
    log_path = LOG_DIR / f"trades_{today}.jsonl"
    entry = {
        "time"     : datetime.now().isoformat(),
        "action"   : action,
        "tf"       : tf,
        "direction": direction,
        "price"    : price,
        "sl"       : sl,
        "tp"       : tp,
        "conf"     : conf,
        "ticket"   : ticket,
        "lot"      : LOT_SIZE,
        "note"     : note,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ════════════════════════════════════════════════════════════
#  STATUS DISPLAY
# ════════════════════════════════════════════════════════════

def show_status():
    """แสดง open orders ปัจจุบันของ bot"""
    orders = get_open_orders()
    bid, ask = get_current_price()

    print(f"\n{'═'*60}")
    print(f"  Gold Auto Trader — Status  |  {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"  Symbol: {SYMBOL}  |  Bid: {bid:.2f}  Ask: {ask:.2f}")
    print(f"  Magic: {MAGIC}  |  Min Conf: {MIN_CONF}%  |  Lot: {LOT_SIZE}")
    print(f"{'═'*60}")

    if not orders:
        print("  ไม่มี open orders")
    else:
        print(f"  Open orders: {len(orders)}/{MAX_ORDERS}")
        print(f"  {'Ticket':<10} {'Type':<6} {'Price':<10} {'SL':<10} "
              f"{'TP':<10} {'Profit':>8} {'Comment'}")
        print(f"  {'-'*70}")
        total_profit = 0.0
        for p in orders:
            typ    = "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL"
            profit = p.profit
            total_profit += profit
            print(f"  #{p.ticket:<9} {typ:<6} {p.price_open:<10.2f} "
                  f"{p.sl:<10.2f} {p.tp:<10.2f} "
                  f"{profit:>+8.2f}  {p.comment}")
        print(f"  {'-'*70}")
        print(f"  Total floating P/L: {total_profit:+.2f} USD")
    print(f"{'═'*60}\n")


# ════════════════════════════════════════════════════════════
#  MAIN BOT LOGIC
# ════════════════════════════════════════════════════════════

def run_once():
    """
    รัน 1 รอบ: อ่าน signals → ตัดสินใจ → execute
    """
    now = datetime.now().strftime("%H:%M:%S")
    log.info(f"{'─'*50}")
    log.info(f"รอบใหม่ {now}")

    # ── 1. โหลด signals ──────────────────────────────────
    signals = load_signals()
    if not signals:
        log.info("ไม่มี signals — รอรอบถัดไป")
        return

    # ── 2. ตรวจ open orders ปัจจุบัน ─────────────────────
    open_orders = get_open_orders()
    total_open  = len(open_orders)
    log.info(f"Open orders: {total_open}/{MAX_ORDERS}")

    # ── 3. วนทุก TF ──────────────────────────────────────
    for tf in TIMEFRAMES:
        sig_raw = signals.get(tf)
        if not sig_raw:
            continue

        parsed = parse_signal(tf, sig_raw)
        if parsed is None:
            continue

        log.info(
            f"[{tf}] signal: {parsed['direction']} | "
            f"conf: {parsed['conf']:.1f}% | "
            f"entry: {parsed['entry']:.2f} | "
            f"SL: {parsed['sl']:.2f} | TP: {parsed['tp']:.2f}"
        )

        # ── safety: ห้ามเกิน MAX_ORDERS รวม ────────────
        if total_open >= MAX_ORDERS:
            log.warning(f"[{tf}] ถึง MAX_ORDERS ({MAX_ORDERS}) แล้ว — ข้าม")
            break

        # ── pyramid: ไม่มีข้อจำกัด direction เดิม ───────
        # เปิดได้เรื่อยๆ ถ้า conf ผ่าน และยังไม่เกิน MAX_ORDERS
        success = open_order(parsed)
        if success:
            total_open += 1
            # refresh open orders หลังเปิดสำเร็จ
            open_orders = get_open_orders()

    log.info(f"จบรอบ | Open orders ตอนนี้: {len(get_open_orders())}")


# ─── Main Loop ────────────────────────────────────────────

def main_loop():
    """รัน bot loop ทุก 1 นาที"""
    log.info("🚀 Gold Auto Trader เริ่มทำงาน")
    log.info(f"   Symbol    : {SYMBOL}")
    log.info(f"   Lot size  : {LOT_SIZE}")
    log.info(f"   Min conf  : {MIN_CONF}%")
    log.info(f"   Max orders: {MAX_ORDERS}")
    log.info(f"   Magic     : {MAGIC}")
    log.info(f"   TFs       : {TIMEFRAMES}")

    if not connect_mt5():
        log.error("เชื่อมต่อ MT5 ไม่ได้ — หยุด")
        return

    show_status()

    # รันทันทีรอบแรก
    run_once()

    # schedule ทุก 1 นาที
    schedule.every(1).minutes.do(run_once)

    log.info("⏳ รอ schedule... (Ctrl+C เพื่อหยุด)")
    try:
        while True:
            schedule.run_pending()
            time.sleep(5)
    except KeyboardInterrupt:
        log.info("🛑 หยุด bot แล้ว")
    finally:
        mt5.shutdown()
        log.info("ปิดการเชื่อมต่อ MT5 แล้ว")


# ════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gold Auto Trader Bot")
    parser.add_argument("--once",     action="store_true",
                        help="รันครั้งเดียวแล้วออก")
    parser.add_argument("--status",   action="store_true",
                        help="แสดง open orders แล้วออก")
    parser.add_argument("--closeall", action="store_true",
                        help="ปิด orders ทั้งหมดของ bot")
    args = parser.parse_args()

    if args.status or args.closeall:
        if not connect_mt5():
            log.error("เชื่อมต่อ MT5 ไม่ได้")
            exit(1)
        if args.status:
            show_status()
        if args.closeall:
            close_all_bot_orders()
            show_status()
        mt5.shutdown()

    elif args.once:
        if not connect_mt5():
            log.error("เชื่อมต่อ MT5 ไม่ได้")
            exit(1)
        run_once()
        show_status()
        mt5.shutdown()

    else:
        main_loop()
