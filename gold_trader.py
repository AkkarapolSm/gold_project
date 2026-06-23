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
   TRADE_LOT=0.01           # lot size คงที่ (ใช้เมื่อ TRADE_RISK_PCT=0)
   TRADE_MIN_CONF=80.0      # confidence ขั้นต่ำ (default 80%)
   TRADE_MAX_ORDERS=5       # max open orders รวมทุก TF (default 5)
   TRADE_MAGIC=20250323     # magic number สำหรับ identify orders

   # ── Risk management (A) ──
   TRADE_RISK_PCT=0               # % เสี่ยงต่อไม้ (>0 = คิด lot จาก SL, 0 = ใช้ TRADE_LOT คงที่)
   TRADE_DAILY_LOSS_LIMIT_PCT=5   # เพดานขาดทุนต่อวัน (% ของ balance ต้นวัน, 0=ปิด)
   TRADE_MAX_DRAWDOWN_PCT=10      # เพดาน drawdown จาก peak equity (%, 0=ปิด)
   TRADE_CLOSE_ON_HALT=false      # true = ปิดออเดอร์ทั้งหมดเมื่อโดน halt
   TRADE_TICK_MAX_AGE=120         # tick เก่าเกินกี่วินาที = ถือว่าตลาดปิด

   # ── Order management (B) ──
   TRADE_MAX_SPREAD=0.50          # B3: spread สูงสุด (USD) ที่ยอมเข้า, 0=ปิด
   TRADE_CONFLICT_GUARD=true      # B2: รวมหลาย TF + กันเปิดสวน
   TRADE_CONFLICT_MARGIN=5.0      # B2: ผลรวม conf ต่างกัน <= นี้ = สูสี งดเทรด
   TRADE_MANAGE_EXITS=true        # B1: เปิดการจัดการ exit (master switch)
   TRADE_BREAKEVEN=true           # B1: ดัน SL มา breakeven เมื่อกำไรพอ
   TRADE_TRAILING=true            # B1: trailing stop
   TRADE_PARTIAL_TP=false         # B1: ปิดบางส่วนเมื่อถึงครึ่งทาง TP

 วิธีใช้:
   python gold_trader.py             # รัน bot loop ทุก 1 นาที
   python gold_trader.py --once      # รันครั้งเดียวแล้วออก
   python gold_trader.py --status    # ดู open orders + สถานะความเสี่ยง
   python gold_trader.py --closeall  # ปิด orders ทั้งหมดของ bot
   python gold_trader.py --reset-risk# ล้างสถานะความเสี่ยง (ปลด halt)
   python gold_trader.py --report daily   # ส่งสรุป P/L รายวันเข้า Telegram
   python gold_trader.py --report weekly  # ส่งสรุป P/L รายสัปดาห์

 หมายเหตุ:
   - bot จะอ่าน signal จาก gold_data/realtime_signals.json
     ที่ถูก generate โดย gold_server.py (background loop)
   - Pyramid: เปิด order ซ้อนได้ ถ้า signal ใหม่ conf >= MIN_CONF
   - TP/SL: ใช้ค่าจาก signal (ATR-based) โดยตรง
   - MAX_ORDERS: safety limit กันเปิดมากเกินไป
   - Risk guard: หยุดเปิดออเดอร์ใหม่อัตโนมัติเมื่อถึงเพดานขาดทุน/drawdown
   - Market guard: ข้ามรอบเมื่อ tick ไม่สด (ตลาดปิด) กัน order ถูกปฏิเสธ
=============================================================
"""

import argparse
import json
import math
import os
import time
import schedule
import logging
from datetime import datetime, timedelta
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd
from dotenv import load_dotenv

from gold_risk import RiskManager

load_dotenv()


def _env_bool(name: str, default: str = "false") -> bool:
    """อ่าน env เป็น boolean (1/true/yes/on = True)"""
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


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

# ── Risk management (A) ──────────────────────────────────
RISK_PCT       = float(os.getenv("TRADE_RISK_PCT",             "0"))    # 0 = ใช้ LOT_SIZE คงที่
DAILY_LOSS_PCT = float(os.getenv("TRADE_DAILY_LOSS_LIMIT_PCT", "5.0"))
MAX_DD_PCT     = float(os.getenv("TRADE_MAX_DRAWDOWN_PCT",     "10.0"))
CLOSE_ON_HALT  = _env_bool("TRADE_CLOSE_ON_HALT", "false")
TICK_MAX_AGE   = int(  os.getenv("TRADE_TICK_MAX_AGE",         "120"))  # วินาที
RISK_STATE     = DATA_DIR / "risk_state.json"

# ── Order management (B) ─────────────────────────────────
# B3 — spread guard
MAX_SPREAD        = float(os.getenv("TRADE_MAX_SPREAD",         "0.50"))  # USD, 0 = ปิด
# B2 — conflict guard (กันหลาย TF ทิศตรงข้าม + กันเปิดสวนของที่ถืออยู่)
CONFLICT_GUARD    = _env_bool("TRADE_CONFLICT_GUARD", "true")
CONFLICT_MARGIN   = float(os.getenv("TRADE_CONFLICT_MARGIN",    "5.0"))   # ผลรวม conf ต่างกัน <= นี้ = สูสี งดเทรด
# B4 — anti-pyramiding (กันกองไม้ทิศเดียว/ราคาเดียว)
MAX_PER_DIR       = int(  os.getenv("TRADE_MAX_PER_DIR",        "2"))     # ไม้ทิศเดียวสูงสุด (0 = ไม่จำกัด)
ENTRY_COOLDOWN    = int(  os.getenv("TRADE_ENTRY_COOLDOWN_SEC", "600"))   # เว้นเวลาเปิด (วินาที, 0 = ปิด)
MIN_GAP_ATR       = float(os.getenv("TRADE_MIN_GAP_ATR",        "1.0"))   # เว้นระยะราคา × ATR (0 = ปิด)
# B5 — basket mode (1 สัญญาณ → เปิดหลายไม้ TP/SL คงที่; ข้าม guard B4)
BASKET_MODE       = _env_bool("TRADE_BASKET_MODE", "false")
BASKET_SIZE       = int(  os.getenv("TRADE_BASKET_SIZE", "10"))
BASKET_TP         = float(os.getenv("TRADE_BASKET_TP",   "2.0"))          # ระยะ TP คงที่ (USD)
BASKET_SL         = float(os.getenv("TRADE_BASKET_SL",   "1.0"))          # ระยะ SL คงที่ (USD)
# B1 — exit management (trailing / breakeven / partial TP)
EXIT_MANAGE       = _env_bool("TRADE_MANAGE_EXITS", "true")               # master switch
BREAKEVEN         = _env_bool("TRADE_BREAKEVEN",     "true")
BE_TRIGGER_FRAC   = float(os.getenv("TRADE_BE_TRIGGER_FRAC",    "0.5"))   # ถึง 50% ทาง TP → ดัน SL มา breakeven
BE_BUFFER_FRAC    = float(os.getenv("TRADE_BE_BUFFER_FRAC",     "0.05"))  # ล็อกกำไรเล็กน้อย (สัดส่วนของระยะ TP)
TRAILING          = _env_bool("TRADE_TRAILING",     "true")
TRAIL_START_FRAC  = float(os.getenv("TRADE_TRAIL_START_FRAC",   "0.5"))   # เริ่ม trail เมื่อถึง % ทาง TP
TRAIL_FRAC        = float(os.getenv("TRADE_TRAIL_FRAC",         "0.5"))   # ระยะ trail = สัดส่วนของระยะ TP
PARTIAL_TP        = _env_bool("TRADE_PARTIAL_TP",   "false")             # ปิดบางส่วน (ปิดไว้ default)
PARTIAL_TP_FRAC   = float(os.getenv("TRADE_PARTIAL_TP_FRAC",    "0.5"))   # ถึง % ทาง TP → ปิดบางส่วน
PARTIAL_CLOSE_PCT = float(os.getenv("TRADE_PARTIAL_CLOSE_PCT",  "50"))    # ปิดกี่ % ของ lot
MANAGED_STATE     = DATA_DIR / "managed_positions.json"

# ── Reporting (C) ────────────────────────────────────────
REPORT_DAILY_TIME  = os.getenv("REPORT_DAILY_TIME",  "23:59").strip()   # "" = ปิด
REPORT_WEEKLY_DAY  = os.getenv("REPORT_WEEKLY_DAY",  "sunday").strip().lower()
REPORT_WEEKLY_TIME = os.getenv("REPORT_WEEKLY_TIME", "23:55").strip()

# ความหมาย retcode ที่พบบ่อย (ช่วย debug ตอน OPEN_FAIL)
RETCODE_MEANINGS = {
    10004: "Requote (ราคาเปลี่ยน)",
    10006: "Request rejected (ถูกปฏิเสธ)",
    10013: "Invalid request (คำขอไม่ถูกต้อง)",
    10014: "Invalid volume (lot ไม่ถูกต้อง)",
    10015: "Invalid price (ราคาไม่ถูกต้อง)",
    10016: "Invalid stops (SL/TP ใกล้/ผิดด้าน)",
    10017: "Trade disabled (ปิดการเทรด)",
    10018: "Market closed (ตลาดปิด)",
    10019: "No money (เงินไม่พอ)",
    10020: "Prices changed (ราคาขยับ)",
    10021: "No quotes (ไม่มีราคา)",
    10024: "Too many requests (ส่งคำขอถี่เกินไป)",
    10026: "Autotrading disabled by server",
    10027: "Autotrading disabled (เปิด Algo Trading ใน MT5)",
    10030: "Unsupported filling mode (เปลี่ยน type_filling)",
    10031: "No connection (ไม่ได้เชื่อมต่อ server)",
    10033: "Pending orders limit reached",
    10034: "Volume limit reached",
}

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

# Risk manager (daily loss limit + max drawdown + auto-halt)
risk = RiskManager(RISK_STATE, daily_loss_pct=DAILY_LOSS_PCT, max_dd_pct=MAX_DD_PCT)

# ใช้ track ว่า tick เดินอยู่ไหม (robust กว่าเทียบ time ตรงๆ เพราะ broker มี offset)
_tick_track = {"time": None, "seen_at": None}

# เวลาเปิดออเดอร์ล่าสุด (สำหรับ cooldown — B4)
_last_entry_ts = 0.0


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


# ── A2: Position sizing ตามทุน ──────────────────────────

def calc_volume(entry: float, sl: float) -> float:
    """
    คำนวณ lot จาก % ความเสี่ยงต่อไม้ + ระยะ SL
        risk_amount  = balance * RISK_PCT%
        loss_per_lot = (|entry-sl| / tick_size) * tick_value
        lot          = risk_amount / loss_per_lot   (ปัดลงตาม volume_step)
    ถ้า RISK_PCT<=0 หรือคำนวณไม่ได้ → คืน LOT_SIZE คงที่
    """
    if RISK_PCT <= 0:
        return LOT_SIZE

    acc  = mt5.account_info()
    info = mt5.symbol_info(SYMBOL)
    if acc is None or info is None:
        log.warning("calc_volume: ไม่มี account/symbol info — ใช้ LOT_SIZE")
        return LOT_SIZE

    sl_dist    = abs(float(entry) - float(sl))
    tick_size  = info.trade_tick_size or info.point
    tick_value = info.trade_tick_value
    if sl_dist <= 0 or not tick_size or not tick_value:
        log.warning("calc_volume: ค่าไม่พร้อม (sl_dist/tick) — ใช้ LOT_SIZE")
        return LOT_SIZE

    risk_amount  = acc.balance * RISK_PCT / 100.0
    loss_per_lot = (sl_dist / tick_size) * tick_value
    if loss_per_lot <= 0:
        return LOT_SIZE

    raw  = risk_amount / loss_per_lot
    step = info.volume_step or 0.01
    lot  = math.floor(raw / step) * step
    lot  = max(info.volume_min, min(lot, info.volume_max))
    lot  = round(lot, 2)

    log.info(
        f"Position sizing | risk {RISK_PCT:.2f}% = {risk_amount:.2f} USD | "
        f"SL dist {sl_dist:.2f} | lot {lot} (raw {raw:.3f})"
    )
    return lot


# ── A3: Market-hours guard ──────────────────────────────

def is_market_open() -> tuple[bool, str]:
    """
    เช็กว่าตลาดเปิด + tick สดพอจะส่งออเดอร์ไหม
    คืน (True, "ok") ถ้าพร้อม / (False, reason) ถ้าไม่พร้อม

    ใช้ 2 ด่าน:
      1) symbol trade_mode ต้องไม่ใช่ DISABLED
      2) tick ต้องเดิน — ถ้า tick.time ค้างนานเกิน TICK_MAX_AGE = ตลาดปิด
         (เทียบแบบ relative กันปัญหา broker server time มี offset)
    """
    info = mt5.symbol_info(SYMBOL)
    if info is None:
        return False, "ไม่มี symbol_info"
    if info.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED:
        return False, "symbol ปิดการเทรด (trade_mode=DISABLED)"

    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None or (tick.bid == 0 and tick.ask == 0):
        return False, "ไม่มี tick / ราคาเป็น 0"

    now = time.time()
    t   = getattr(tick, "time", 0)

    # tick เดิน (เวลาเปลี่ยน) → สด, รีเซ็ตตัวจับเวลา
    if _tick_track["time"] is None or t != _tick_track["time"]:
        _tick_track["time"]    = t
        _tick_track["seen_at"] = now
        return True, "ok"

    # tick.time เท่าเดิม → ดูว่าค้างมานานแค่ไหน
    age = now - (_tick_track["seen_at"] or now)
    if age > TICK_MAX_AGE:
        return False, f"tick ค้าง {int(age)}s (ตลาดน่าจะปิด)"
    return True, "ok"


# ── เปิด Order ───────────────────────────────────────────

def _notify_open_fail(tf: str, direction: str, reason: str,
                      price: float = 0, sl: float = 0, tp: float = 0,
                      conf: float = 0, lot: float = None):
    """C1: log + แจ้ง Telegram เมื่อเปิดออเดอร์ไม่สำเร็จ (พร้อมเหตุผล)"""
    _log_trade("OPEN_FAIL", tf, direction, price, sl, tp, conf, 0,
               note=reason, lot=lot)
    if tg:
        try:
            tg.send_order("OPEN_FAIL", tf, direction, round(price, 2), sl, tp,
                          round(conf, 1), 0, lot=lot or LOT_SIZE, note=reason)
        except Exception as e:
            log.warning(f"Telegram OPEN_FAIL error: {e}")


def open_order(parsed: dict, sl_dist: float = None, tp_dist: float = None) -> bool:
    """
    ส่ง market order ไปที่ MT5
    parsed = { tf, direction, conf, entry, sl, tp }
    sl_dist/tp_dist: ถ้าใส่มา (basket mode) ใช้เป็นระยะ SL/TP คงที่ (USD) จากราคาเข้า
                     แทน SL/TP แบบ ATR ของ signal
    """
    tf        = parsed["tf"]
    direction = parsed["direction"]
    bid, ask  = get_current_price()

    if bid == 0:
        log.error("ไม่สามารถดึงราคาได้ — ข้าม")
        return False

    # B3: spread guard — ข้ามถ้า spread กว้างผิดปกติ (กันเข้าในช่วงผันผวน/ข่าว)
    spread = ask - bid
    if MAX_SPREAD > 0 and spread > MAX_SPREAD:
        log.warning(f"[{tf}] spread {spread:.2f} > {MAX_SPREAD:.2f} USD — ข้าม")
        _log_trade("SKIP_SPREAD", tf, direction, ask, parsed["sl"], parsed["tp"],
                   parsed["conf"], 0, note=f"spread={spread:.2f}")
        return False

    # A2: lot ตาม % ความเสี่ยง (ใช้ entry/SL จาก signal) — fallback เป็น LOT_SIZE
    lot = calc_volume(parsed["entry"], parsed["sl"])

    if "UP" in direction:
        order_type = mt5.ORDER_TYPE_BUY
        price      = ask          # BUY ใช้ ask
    else:
        order_type = mt5.ORDER_TYPE_SELL
        price      = bid          # SELL ใช้ bid

    # SL/TP: basket mode = ระยะคงที่จากราคาเข้า, ไม่งั้นใช้ค่า ATR จาก signal
    if sl_dist is not None and tp_dist is not None:
        if order_type == mt5.ORDER_TYPE_BUY:
            sl = normalize_price(price - sl_dist)
            tp = normalize_price(price + tp_dist)
        else:
            sl = normalize_price(price + sl_dist)
            tp = normalize_price(price - tp_dist)
    else:
        sl = normalize_price(parsed["sl"])
        tp = normalize_price(parsed["tp"])

    comment = f"Gold_{tf}_{direction[:2]}_{int(parsed['conf'])}"

    request = {
        "action"      : mt5.TRADE_ACTION_DEAL,
        "symbol"      : SYMBOL,
        "volume"      : lot,
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
        reason = f"order_send คืน None | {mt5.last_error()}"
        log.error(f"[{tf}] {reason}")
        _notify_open_fail(tf, direction, reason, price, sl, tp, parsed["conf"], lot)
        return False

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        log.info(
            f"✅ เปิด order สำเร็จ | {tf} | {direction} | "
            f"Lot:{lot} | Price:{price:.2f} | spread:{spread:.2f} | "
            f"SL:{sl:.2f} | TP:{tp:.2f} | "
            f"Conf:{parsed['conf']:.1f}% | Ticket:#{result.order}"
        )
        _log_trade("OPEN", tf, direction, price, sl, tp,
                   parsed["conf"], result.order, lot=lot)
        if tg:
            try:
                tg.send_order("OPEN", tf, direction, round(price, 2),
                              sl, tp, round(parsed["conf"], 1),
                              result.order, lot=lot)
            except Exception as e:
                log.warning(f"Telegram order error: {e}")
        return True
    else:
        meaning = RETCODE_MEANINGS.get(result.retcode, "")
        reason  = f"retcode={result.retcode} {meaning} | {result.comment}".strip()
        log.error(f"❌ เปิด order ล้มเหลว | {tf} | {reason}")
        _notify_open_fail(tf, direction, reason, price, sl, tp, parsed["conf"], lot)
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
                   note=f"profit={profit:+.2f}", lot=position.volume)
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
#  B1 — EXIT MANAGEMENT (breakeven / trailing / partial TP)
# ════════════════════════════════════════════════════════════

_managed = None   # state ของ position ที่จัดการแล้ว (กัน partial ซ้ำ)


def _load_managed() -> dict:
    global _managed
    if _managed is None:
        try:
            _managed = json.loads(MANAGED_STATE.read_text(encoding="utf-8")) \
                       if MANAGED_STATE.exists() else {}
        except Exception:
            _managed = {}
    return _managed


def _save_managed():
    try:
        MANAGED_STATE.parent.mkdir(parents=True, exist_ok=True)
        MANAGED_STATE.write_text(json.dumps(_managed, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    except Exception:
        pass


def modify_sltp(position, new_sl: float = None, new_tp: float = None) -> bool:
    """แก้ SL/TP ของ position ที่เปิดอยู่ (TRADE_ACTION_SLTP)"""
    request = {
        "action"  : mt5.TRADE_ACTION_SLTP,
        "symbol"  : SYMBOL,
        "position": position.ticket,
        "sl"      : normalize_price(new_sl) if new_sl is not None else position.sl,
        "tp"      : normalize_price(new_tp) if new_tp is not None else position.tp,
        "magic"   : MAGIC,
    }
    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        return True
    retcode = result.retcode if result else "None"
    log.warning(f"แก้ SL/TP #{position.ticket} ไม่สำเร็จ: retcode={retcode}")
    return False


def close_partial(position, volume: float) -> bool:
    """ปิดบางส่วนของ position ด้วย market order"""
    info = mt5.symbol_info(SYMBOL)
    step = info.volume_step or 0.01
    vmin = info.volume_min or 0.01

    vol = round(math.floor(volume / step) * step, 2)
    if vol < vmin:
        return False
    # ถ้าปิดแล้วส่วนที่เหลือเล็กกว่า min → ปิดทั้งไม้ไปเลย
    if round(position.volume - vol, 2) < vmin:
        vol = position.volume

    bid, ask = get_current_price()
    if position.type == mt5.ORDER_TYPE_BUY:
        close_type, price = mt5.ORDER_TYPE_SELL, bid
    else:
        close_type, price = mt5.ORDER_TYPE_BUY, ask

    request = {
        "action"      : mt5.TRADE_ACTION_DEAL,
        "symbol"      : SYMBOL,
        "volume"      : vol,
        "type"        : close_type,
        "position"    : position.ticket,
        "price"       : price,
        "deviation"   : 20,
        "magic"       : MAGIC,
        "comment"     : f"partial_{position.ticket}",
        "type_time"   : mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        log.info(f"✂️ Partial close #{position.ticket} | {vol} lot @ {price:.2f}")
        _log_trade("PARTIAL_CLOSE", "-", "CLOSE", price, 0, 0, 0,
                   position.ticket, note=f"vol={vol}", lot=vol)
        if tg:
            try:
                tg._send(tg.order_chat,
                         f"✂️ Partial close #{position.ticket}  {vol} lot @ {price:.2f}")
            except Exception:
                pass
        return True
    retcode = result.retcode if result else "None"
    log.warning(f"Partial close #{position.ticket} ไม่สำเร็จ: retcode={retcode}")
    return False


def manage_open_positions():
    """
    B1: จัดการ exit ของออเดอร์ที่เปิดอยู่ทุกรอบ
       - partial TP : ถึง PARTIAL_TP_FRAC ของทาง TP → ปิดบางส่วน (ครั้งเดียว)
       - breakeven  : ถึง BE_TRIGGER_FRAC → ดัน SL มาที่ทุน (+buffer)
       - trailing   : ถึง TRAIL_START_FRAC → เลื่อน SL ตามราคา (เฉพาะทางที่ดีขึ้น)
    SL จะถูกแก้เฉพาะเมื่อ "แน่นขึ้น" เท่านั้น (ไม่ผ่อนคลายความเสี่ยง)
    """
    positions = get_open_orders()
    state = _load_managed()

    # cleanup state ของไม้ที่ปิดไปแล้ว
    open_tickets = {str(p.ticket) for p in positions}
    for k in list(state.keys()):
        if k not in open_tickets:
            del state[k]

    if not positions:
        _save_managed()
        return

    bid, ask = get_current_price()
    if bid == 0:
        return

    info  = mt5.symbol_info(SYMBOL)
    point = info.point or 0.01
    stops = (info.trade_stops_level or 0) * point   # ระยะ SL ขั้นต่ำจากราคา

    for p in positions:
        key   = str(p.ticket)
        st    = state.setdefault(key, {"partial_done": False})
        is_buy = p.type == mt5.ORDER_TYPE_BUY
        entry = p.price_open
        tp    = p.tp
        cur   = bid if is_buy else ask        # ราคาที่จะใช้ exit

        if not tp or abs(tp - entry) <= 0:
            continue
        tp_dist  = abs(tp - entry)
        progress = ((cur - entry) if is_buy else (entry - cur)) / tp_dist

        # ── partial TP ──
        if PARTIAL_TP and progress >= PARTIAL_TP_FRAC and not st.get("partial_done"):
            if close_partial(p, p.volume * PARTIAL_CLOSE_PCT / 100.0):
                st["partial_done"] = True
                _save_managed()

        # ── คำนวณ SL ใหม่ (breakeven + trailing, เลือกที่แน่นกว่า) ──
        new_sl = None
        if BREAKEVEN and progress >= BE_TRIGGER_FRAC:
            buf    = BE_BUFFER_FRAC * tp_dist
            new_sl = entry + buf if is_buy else entry - buf
        if TRAILING and progress >= TRAIL_START_FRAC:
            td    = TRAIL_FRAC * tp_dist
            trail = cur - td if is_buy else cur + td
            if new_sl is None:
                new_sl = trail
            else:
                new_sl = max(new_sl, trail) if is_buy else min(new_sl, trail)

        if new_sl is None:
            continue

        # เคารพระยะ SL ขั้นต่ำของโบรกเกอร์ + แก้เฉพาะตอนที่แน่นขึ้น
        if is_buy:
            new_sl   = min(new_sl, cur - stops)
            improved = new_sl > 0 and new_sl > (p.sl + point)
        else:
            new_sl   = max(new_sl, cur + stops)
            improved = new_sl > 0 and (p.sl <= 0 or new_sl < (p.sl - point))

        if improved and modify_sltp(p, new_sl=new_sl):
            log.info(f"🛡️ เลื่อน SL #{p.ticket} → {normalize_price(new_sl):.2f} "
                     f"(progress {progress*100:.0f}%)")

    _save_managed()


# ════════════════════════════════════════════════════════════
#  TRADE LOGGER
# ════════════════════════════════════════════════════════════

def _log_trade(action: str, tf: str, direction: str,
               price: float, sl: float, tp: float,
               conf: float, ticket: int, note: str = "", lot: float = None):
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
        "lot"      : lot if lot is not None else LOT_SIZE,
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

    sizing = f"risk {RISK_PCT:.2f}%/ไม้" if RISK_PCT > 0 else f"คงที่ {LOT_SIZE}"

    print(f"\n{'═'*60}")
    print(f"  Gold Auto Trader — Status  |  {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"  Symbol: {SYMBOL}  |  Bid: {bid:.2f}  Ask: {ask:.2f}")
    print(f"  Magic: {MAGIC}  |  Min Conf: {MIN_CONF}%  |  Lot: {sizing}")

    # ── Risk guard status ──
    acc = mt5.account_info()
    if acc is not None and risk.enabled:
        res = risk.update(acc.balance, acc.equity)
        state = "🛑 HALTED" if res["halted"] else "✅ OK"
        print(f"  Risk : {state}  |  DayP/L {res['daily_pl']:+.2f} "
              f"({res['daily_pl_pct']:+.2f}%)  |  DD {res['drawdown_pct']:.2f}%")
        print(f"         เพดาน: ขาดทุนวัน {DAILY_LOSS_PCT:.1f}%  ·  drawdown {MAX_DD_PCT:.1f}%")
        if res["halted"]:
            print(f"         เหตุผล: {res['halt_reason']}")
    elif acc is not None:
        print(f"  Risk : ปิดการเช็ค (TRADE_DAILY_LOSS_LIMIT_PCT/MAX_DRAWDOWN_PCT = 0)")

    # ── Order management (B) ──
    exits = []
    if EXIT_MANAGE:
        if BREAKEVEN: exits.append("breakeven")
        if TRAILING:  exits.append("trailing")
        if PARTIAL_TP: exits.append("partialTP")
    exits_str = ", ".join(exits) if exits else "ปิด"
    print(f"  Order: spread<={MAX_SPREAD:.2f} · conflict-guard {'on' if CONFLICT_GUARD else 'off'} · "
          f"exits[{exits_str}]")
    if BASKET_MODE:
        print(f"  Basket: ON · {BASKET_SIZE} ไม้/สัญญาณ · TP {BASKET_TP} / SL {BASKET_SL} USD (ข้าม guard กองไม้)")
    else:
        print(f"  Pyrmd: max/dir {MAX_PER_DIR} · cooldown {ENTRY_COOLDOWN}s · gap {MIN_GAP_ATR}×ATR")
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
#  C2 — P/L REPORTING (daily / weekly → Telegram)
# ════════════════════════════════════════════════════════════

def summarize_deals(from_dt: datetime, to_dt: datetime) -> dict:
    """
    สรุปผลการเทรดของบอท (filter magic+symbol) จาก history deals ของ MT5
    นับเป็น "ไม้" ตาม position_id (รวม partial เข้าด้วยกัน)
    """
    empty = {"trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
             "pnl": 0.0, "gross_win": 0.0, "gross_loss": 0.0,
             "profit_factor": 0.0, "best": 0.0, "worst": 0.0}

    deals = mt5.history_deals_get(from_dt, to_dt)
    if deals is None:
        return empty

    mine = [d for d in deals if d.magic == MAGIC and d.symbol == SYMBOL]
    if not mine:
        return empty

    def net(d):
        return d.profit + d.swap + d.commission

    # รวม P/L ต่อ position (เฉพาะขา OUT = ปิด)
    pos_pnl = {}
    for d in mine:
        if d.entry == mt5.DEAL_ENTRY_OUT:
            pos_pnl[d.position_id] = pos_pnl.get(d.position_id, 0.0) + net(d)

    pnls   = list(pos_pnl.values())
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    n      = len(pnls)

    gross_win  = sum(wins)
    gross_loss = -sum(losses)
    pf = (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)

    return {
        "trades"       : n,
        "wins"         : len(wins),
        "losses"       : len(losses),
        "win_rate"     : (len(wins) / n * 100) if n else 0.0,
        "pnl"          : sum(net(d) for d in mine),
        "gross_win"    : gross_win,
        "gross_loss"   : gross_loss,
        "profit_factor": pf,
        "best"         : max(pnls) if pnls else 0.0,
        "worst"        : min(pnls) if pnls else 0.0,
    }


def report_pnl(period: str = "daily") -> dict:
    """สร้างสรุป P/L แล้วส่งเข้า Telegram (ช่อง order)"""
    now = datetime.now()
    if period == "weekly":
        start = now - timedelta(days=7)
        title = "สรุป P/L รายสัปดาห์ (7 วัน)"
    else:
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        title = "สรุป P/L รายวัน"

    s   = summarize_deals(start, now)
    acc = mt5.account_info()
    bal = acc.balance if acc else 0.0
    eq  = acc.equity  if acc else 0.0
    pf  = s["profit_factor"]
    pf_str = "∞" if pf == float("inf") else f"{pf:.2f}"

    log.info(f"[REPORT {period}] {s['trades']} ไม้ | win {s['win_rate']:.1f}% | "
             f"P/L {s['pnl']:+.2f} | PF {pf_str}")

    msg = (
        f"📊 {title}\n"
        f"{start.strftime('%d/%m')} – {now.strftime('%d/%m %H:%M')}\n"
        "──────────────\n"
        f"ปิดไปแล้ว : {s['trades']} ไม้  (W {s['wins']} / L {s['losses']})\n"
        f"Win rate  : {s['win_rate']:.1f}%\n"
        f"Realized  : {s['pnl']:+.2f} USD\n"
        f"Profit factor : {pf_str}\n"
        f"ดีสุด/แย่สุด : {s['best']:+.2f} / {s['worst']:+.2f}\n"
        "──────────────\n"
        f"Balance : {bal:.2f}  |  Equity : {eq:.2f} USD"
    )
    if tg:
        try:
            tg._send(tg.order_chat, msg)
        except Exception as e:
            log.warning(f"Telegram report error: {e}")
    return s


# ════════════════════════════════════════════════════════════
#  MAIN BOT LOGIC
# ════════════════════════════════════════════════════════════

def _alert_halt(res: dict):
    """ส่ง Telegram แจ้งเตือนเมื่อ risk guard สั่งหยุดเทรด (ส่งครั้งเดียว)"""
    if not tg:
        return
    kind = "ขาดทุนรายวัน" if res["halt_kind"] == "DAILY_LOSS" else "Max Drawdown"
    msg = (
        f"🛑 หยุดเทรดอัตโนมัติ — {kind}\n"
        f"{res['halt_reason']}\n"
        "──────────────\n"
        f"Balance : {res['balance']:.2f} USD\n"
        f"Equity  : {res['equity']:.2f} USD\n"
        f"Day P/L : {res['daily_pl']:+.2f} ({res['daily_pl_pct']:+.2f}%)\n"
        f"Drawdown: {res['drawdown_pct']:.2f}%\n"
        f"เวลา    : {datetime.now().strftime('%d/%m %H:%M:%S')}"
    )
    try:
        tg._send(tg.order_chat, msg)
    except Exception as e:
        log.warning(f"Telegram halt alert error: {e}")


def check_risk() -> bool:
    """
    A1: เช็กเพดานความเสี่ยง (daily loss / drawdown)
    คืน True = เทรดต่อได้ / False = โดน halt (ห้ามเปิดออเดอร์ใหม่)
    """
    if not risk.enabled:
        return True

    acc = mt5.account_info()
    if acc is None:
        log.warning("ดึง account_info ไม่ได้ — ข้ามการเช็ค risk รอบนี้")
        return True

    res = risk.update(acc.balance, acc.equity)
    log.info(
        f"Risk | Bal {acc.balance:.2f} Eq {acc.equity:.2f} | "
        f"DayP/L {res['daily_pl']:+.2f} ({res['daily_pl_pct']:+.2f}%) | "
        f"DD {res['drawdown_pct']:.2f}%"
    )

    if res["allowed"]:
        return True

    log.warning(f"🛑 STOP เทรด: {res['halt_reason']}")
    if res["just_halted"] and not risk.alerted:
        _alert_halt(res)
        risk.mark_alerted()
        if CLOSE_ON_HALT:
            log.warning("TRADE_CLOSE_ON_HALT=on → ปิดออเดอร์ทั้งหมด")
            close_all_bot_orders()
    return False


def _net_exposure(open_orders: list) -> str | None:
    """ทิศทางสุทธิของออเดอร์ที่ถืออยู่ — 'UP' / 'DOWN' / None"""
    buys  = sum(1 for p in open_orders if p.type == mt5.ORDER_TYPE_BUY)
    sells = sum(1 for p in open_orders if p.type == mt5.ORDER_TYPE_SELL)
    if buys > sells:
        return "UP"
    if sells > buys:
        return "DOWN"
    return None


def resolve_conflict(candidates: list, open_orders: list) -> tuple[str | None, str]:
    """
    B2: รวมสัญญาณหลาย TF เป็นทิศทางเดียว + กันเปิดสวนของที่ถืออยู่
    คืน (allowed_dir, reason):
       allowed_dir = "UP"/"DOWN" = เปิดได้เฉพาะทางนี้
                     "ANY"       = guard ปิดอยู่ เปิดได้ทุกทาง
                     None        = งดเปิดรอบนี้ (สูสี/สวนของเดิม)
    """
    if not CONFLICT_GUARD:
        return "ANY", ""
    if not candidates:
        return None, ""

    ups = [c for c in candidates if "UP" in c["direction"]]
    dns = [c for c in candidates if "UP" not in c["direction"]]
    up_conf = sum(c["conf"] for c in ups)
    dn_conf = sum(c["conf"] for c in dns)

    reason = ""
    if ups and dns:
        if abs(up_conf - dn_conf) <= CONFLICT_MARGIN:
            return None, (f"TF ขัดกันและสูสี (UP {len(ups)}TF/{up_conf:.0f} vs "
                          f"DN {len(dns)}TF/{dn_conf:.0f}) — งดเทรด")
        winner = "UP" if up_conf > dn_conf else "DOWN"
        reason = (f"TF ขัดกัน → เลือก {winner} "
                  f"(UP {up_conf:.0f} vs DN {dn_conf:.0f})")
    else:
        winner = "UP" if ups else "DOWN"

    # กันเปิดสวนกับโพสิชันที่ถืออยู่
    net = _net_exposure(open_orders)
    if net and winner != net:
        return None, (f"signal {winner} สวนกับโพสิชันที่ถืออยู่ ({net}) — งดเปิดสวน")

    return winner, reason


def _pyramid_block(parsed: dict, cand_dir: str, open_orders: list) -> str | None:
    """
    B4: กันกองไม้สวนเทรนด์ — คืนเหตุผลถ้าควร "ข้าม" / None ถ้าเปิดได้
       - per-direction cap : ไม้ทิศเดียวกันห้ามเกิน MAX_PER_DIR
       - cooldown          : เว้นเวลาขั้นต่ำหลังเปิดไม้ล่าสุด
       - price separation  : ห้ามเปิดถ้าราคาใกล้ไม้เดิมทิศเดียว < MIN_GAP_ATR × ATR
    """
    want_type = mt5.ORDER_TYPE_BUY if cand_dir == "UP" else mt5.ORDER_TYPE_SELL
    same = [p for p in open_orders if p.type == want_type]

    if MAX_PER_DIR > 0 and len(same) >= MAX_PER_DIR:
        return f"ถึงเพดานไม้ทิศ {cand_dir} ({len(same)}/{MAX_PER_DIR})"

    if ENTRY_COOLDOWN > 0:
        gap = time.time() - _last_entry_ts
        if gap < ENTRY_COOLDOWN:
            return f"cooldown เหลือ {int(ENTRY_COOLDOWN - gap)}s"

    if MIN_GAP_ATR > 0 and same:
        atr_est = abs(parsed["entry"] - parsed["sl"]) / 1.5   # ประมาณ ATR จาก SL ของ signal
        if atr_est > 0:
            nearest = min(abs(parsed["entry"] - p.price_open) for p in same)
            if nearest < MIN_GAP_ATR * atr_est:
                return (f"ราคาใกล้ไม้เดิม {nearest:.2f} < {MIN_GAP_ATR}×ATR "
                        f"({MIN_GAP_ATR * atr_est:.2f}) — กันกองราคาเดียว")
    return None


def run_once():
    """
    รัน 1 รอบ: เช็ค risk/ตลาด → จัดการ exit → อ่าน signals → ตัดสินใจ → execute
    """
    global _last_entry_ts
    now = datetime.now().strftime("%H:%M:%S")
    log.info(f"{'─'*50}")
    log.info(f"รอบใหม่ {now}")

    # ── 0a. A3: market-hours guard (tick ต้องสด) ─────────
    is_open, why = is_market_open()
    if not is_open:
        log.info(f"⛔ ข้ามรอบนี้ — {why}")
        return

    # ── 0b. B1: จัดการ exit ของไม้ที่เปิดอยู่ (ทำทุกรอบ) ──
    if EXIT_MANAGE:
        manage_open_positions()

    # ── 0c. A1: risk guard (daily loss / drawdown) ───────
    if not check_risk():
        return

    # ── 1. โหลด signals ──────────────────────────────────
    signals = load_signals()
    if not signals:
        log.info("ไม่มี signals — รอรอบถัดไป")
        return

    # ── 2. ตรวจ open orders ปัจจุบัน ─────────────────────
    open_orders = get_open_orders()
    total_open  = len(open_orders)
    log.info(f"Open orders: {total_open}/{MAX_ORDERS}")

    # ── 3. รวบรวม candidates ที่ผ่านเกณฑ์ ────────────────
    candidates = []
    for tf in TIMEFRAMES:
        parsed = parse_signal(tf, signals.get(tf))
        if parsed is None:
            continue
        log.info(
            f"[{tf}] signal: {parsed['direction']} | "
            f"conf: {parsed['conf']:.1f}% | "
            f"entry: {parsed['entry']:.2f} | "
            f"SL: {parsed['sl']:.2f} | TP: {parsed['tp']:.2f}"
        )
        candidates.append(parsed)

    if not candidates:
        log.info(f"จบรอบ | ไม่มี signal ผ่านเกณฑ์ | Open: {len(get_open_orders())}")
        return

    # ── 4. B2: ตัดสินทิศทางรวม + กันสัญญาณขัดกัน ─────────
    allowed_dir, reason = resolve_conflict(candidates, open_orders)
    if reason:
        log.info(f"Conflict guard: {reason}")
    if allowed_dir is None:
        log.info(f"จบรอบ | งดเปิดออเดอร์ (conflict guard) | Open: {len(get_open_orders())}")
        return

    # ── 5. วนเปิดเฉพาะทางที่อนุญาต ───────────────────────
    for parsed in candidates:
        cand_dir = "UP" if "UP" in parsed["direction"] else "DOWN"
        if allowed_dir != "ANY" and cand_dir != allowed_dir:
            log.info(f"[{parsed['tf']}] {parsed['direction']} ไม่ตรงทางที่อนุญาต "
                     f"({allowed_dir}) — ข้าม")
            continue

        # ── B5: basket mode — 1 สัญญาณเปิดหลายไม้ TP/SL คงที่ (ข้าม guard B4) ──
        if BASKET_MODE:
            opened = 0
            for _ in range(BASKET_SIZE):
                if total_open >= MAX_ORDERS:
                    log.warning(f"[{parsed['tf']}] ถึง MAX_ORDERS ({MAX_ORDERS}) — หยุด basket")
                    break
                if open_order(parsed, sl_dist=BASKET_SL, tp_dist=BASKET_TP):
                    total_open += 1
                    opened     += 1
                    open_orders = get_open_orders()
            if opened:
                _last_entry_ts = time.time()
            log.info(f"[{parsed['tf']}] basket: เปิด {opened}/{BASKET_SIZE} ไม้ "
                     f"(TP {BASKET_TP} / SL {BASKET_SL} USD)")
            continue

        # ── safety: ห้ามเกิน MAX_ORDERS รวม ────────────
        if total_open >= MAX_ORDERS:
            log.warning(f"[{parsed['tf']}] ถึง MAX_ORDERS ({MAX_ORDERS}) แล้ว — ข้าม")
            break

        # ── B4: กันกองไม้สวนเทรนด์ (per-dir cap / cooldown / price gap) ──
        block = _pyramid_block(parsed, cand_dir, open_orders)
        if block:
            log.info(f"[{parsed['tf']}] ข้าม — {block}")
            continue

        # ── pyramid: เปิดซ้อนทางเดียวกันได้ ถ้ายังไม่เกิน MAX_ORDERS ──
        if open_order(parsed):
            total_open    += 1
            _last_entry_ts = time.time()
            open_orders    = get_open_orders()

    log.info(f"จบรอบ | Open orders ตอนนี้: {len(get_open_orders())}")


# ─── Main Loop ────────────────────────────────────────────

def main_loop():
    """รัน bot loop ทุก 1 นาที"""
    log.info("🚀 Gold Auto Trader เริ่มทำงาน")
    log.info(f"   Symbol    : {SYMBOL}")
    log.info(f"   Lot size  : {'risk %.2f%%/ไม้' % RISK_PCT if RISK_PCT > 0 else LOT_SIZE}")
    log.info(f"   Min conf  : {MIN_CONF}%")
    log.info(f"   Max orders: {MAX_ORDERS}")
    log.info(f"   Magic     : {MAGIC}")
    log.info(f"   TFs       : {TIMEFRAMES}")
    log.info(f"   Risk grd  : daily {DAILY_LOSS_PCT}% · DD {MAX_DD_PCT}%")
    log.info(f"   Spread max: {MAX_SPREAD} · Conflict guard: {CONFLICT_GUARD}")
    log.info(f"   Exit mgmt : manage={EXIT_MANAGE} BE={BREAKEVEN} trail={TRAILING} partial={PARTIAL_TP}")
    if BASKET_MODE:
        log.info(f"   Basket    : ON · {BASKET_SIZE} ไม้/สัญญาณ · TP {BASKET_TP} / SL {BASKET_SL} USD")
    else:
        log.info(f"   Anti-pyr  : max/dir {MAX_PER_DIR} · cooldown {ENTRY_COOLDOWN}s · gap {MIN_GAP_ATR}xATR")

    if not connect_mt5():
        log.error("เชื่อมต่อ MT5 ไม่ได้ — หยุด")
        return

    show_status()

    # รันทันทีรอบแรก
    run_once()

    # schedule ทุก 1 นาที
    schedule.every(1).minutes.do(run_once)

    # ── C2: รายงาน P/L รายวัน / รายสัปดาห์ เข้า Telegram ──
    if REPORT_DAILY_TIME:
        schedule.every().day.at(REPORT_DAILY_TIME).do(report_pnl, period="daily")
        log.info(f"   Report    : รายวัน {REPORT_DAILY_TIME}")
    weekly = getattr(schedule.every(), REPORT_WEEKLY_DAY, None)
    if weekly is not None and REPORT_WEEKLY_TIME:
        weekly.at(REPORT_WEEKLY_TIME).do(report_pnl, period="weekly")
        log.info(f"   Report    : รายสัปดาห์ {REPORT_WEEKLY_DAY} {REPORT_WEEKLY_TIME}")

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
    parser.add_argument("--reset-risk", action="store_true",
                        help="ล้างสถานะความเสี่ยง (ปลด halt / รีเซ็ต baseline)")
    parser.add_argument("--report", choices=["daily", "weekly"],
                        help="ส่งสรุป P/L (daily/weekly) เข้า Telegram แล้วออก")
    args = parser.parse_args()

    if args.reset_risk:
        risk.reset()
        log.info("♻️  รีเซ็ตสถานะความเสี่ยงแล้ว (ปลด halt) — "
                 f"ลบ baseline ใน {RISK_STATE.name}")
        exit(0)

    if args.report:
        if not connect_mt5():
            log.error("เชื่อมต่อ MT5 ไม่ได้")
            exit(1)
        report_pnl(period=args.report)
        mt5.shutdown()
        exit(0)

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
