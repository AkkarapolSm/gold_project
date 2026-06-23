"""
=============================================================
 Gold Signal Adjuster (D2) — รวม fundamental / sentiment / regime
 เข้าสู่สัญญาณจริง โดยไม่ต้อง retrain โมเดล
=============================================================
 แนวคิด:
   ML ensemble ให้ทิศทาง + confidence พื้นฐาน → ชั้นนี้เป็น
   "context overlay" ปรับ confidence ขึ้น/ลง ตามบริบทตลาด
   และ veto (เปลี่ยนเป็น NEUTRAL) เมื่อบริบทขัดแย้งรุนแรง

   - Fundamental : fundamentals.json → score.normalized (-100..100)
   - Sentiment   : sentiment.json    → avg_net_score / bull_pct
   - Regime      : detect_regime(df) → TRENDING / SIDEWAYS / VOLATILE

 ไม่แตะ entry/SL/TP — ปรับเฉพาะ confidence (+ veto) ซึ่งเป็นตัว
 ตัดสิน threshold การส่ง Telegram (>=70%) และเข้าออเดอร์ (>=80%)
 จึงปลอดภัยกับโมเดลที่ deploy อยู่ (ไม่เปลี่ยน feature/รูปแบบไฟล์)

 ปรับได้ผ่าน .env (ดู DEFAULT ด้านล่าง)
=============================================================
"""

import json
import os
from copy import deepcopy
from pathlib import Path

DATA_DIR = Path("./gold_data")


def _env_bool(name, default="false"):
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


# ── config (อ่านตอน import) ───────────────────────────────
ENABLED       = _env_bool("SIGNAL_ADJUST", "true")
W_FUND        = float(os.getenv("ADJUST_W_FUND",    "0.5"))   # น้ำหนัก fundamental
W_SENT        = float(os.getenv("ADJUST_W_SENT",    "0.5"))   # น้ำหนัก sentiment
MAX_CTX       = float(os.getenv("ADJUST_MAX_CTX",   "10.0"))  # ปรับ conf สูงสุดจาก fund+sent (จุด %)
REGIME_BONUS  = float(os.getenv("ADJUST_REGIME_BONUS",  "5.0"))   # trend หนุนทิศเดียวกัน
REGIME_PENALTY= float(os.getenv("ADJUST_REGIME_PENALTY","8.0"))   # trend สวนทิศ ML
SIDEWAYS_MULT = float(os.getenv("ADJUST_SIDEWAYS_MULT", "0.90"))  # damp ตอน sideways
VOLATILE_MULT = float(os.getenv("ADJUST_VOLATILE_MULT", "0.85"))  # damp ตอน volatile
VETO          = _env_bool("ADJUST_VETO", "true")
VETO_AGREE    = float(os.getenv("ADJUST_VETO_AGREE", "-0.6"))     # agreement ต่ำกว่านี้ = veto
VETO_REGIME   = _env_bool("ADJUST_VETO_REGIME", "true")           # veto เด็ดขาดเมื่อ trend สวนทิศ ML


def _ml_sign(direction: str) -> int:
    """+1 = UP, -1 = DOWN, 0 = NEUTRAL/อื่นๆ"""
    d = (direction or "").upper()
    if "UP" in d:
        return 1
    if "DOWN" in d or "DN" in d:
        return -1
    return 0


def fund_bias(fund: dict) -> float:
    """bias พื้นฐานต่อทอง จาก fundamentals.json → [-1,1] (บวก=หนุนทอง)"""
    if not fund:
        return 0.0
    norm = fund.get("score", {}).get("normalized", 0)
    try:
        return _clip(float(norm) / 100.0, -1.0, 1.0)
    except (TypeError, ValueError):
        return 0.0


def sent_bias(sent: dict) -> float:
    """bias จาก sentiment ข่าว → [-1,1] (บวก=ข่าวหนุนทอง)"""
    if not sent:
        return 0.0
    s = sent.get("sentiment", sent)   # รองรับทั้งไฟล์เต็มและ dict ย่อย
    score = s.get("avg_net_score", 0)
    try:
        return _clip(float(score) / 3.0, -1.0, 1.0)
    except (TypeError, ValueError):
        return 0.0


def load_context(df=None, tf: str = "M15") -> dict:
    """โหลด fundamental + sentiment จากไฟล์ และ (ถ้ามี df) ตรวจ regime"""
    ctx = {"fund": {}, "sent": {}, "regime": {}}

    fpath = DATA_DIR / "fundamentals.json"
    if fpath.exists():
        try:
            ctx["fund"] = json.loads(fpath.read_text(encoding="utf-8"))
        except Exception:
            pass

    spath = DATA_DIR / "sentiment.json"
    if spath.exists():
        try:
            ctx["sent"] = json.loads(spath.read_text(encoding="utf-8"))
        except Exception:
            pass

    if df is not None:
        try:
            from gold_regime import detect_regime
            ctx["regime"] = detect_regime(df)
        except Exception:
            ctx["regime"] = {}

    return ctx


def adjust_signal(sig: dict, regime: dict = None,
                  fund: dict = None, sent: dict = None) -> dict:
    """
    ปรับ confidence ของ signal ตามบริบท (fund/sent/regime)
    คืน signal ใหม่ (copy) + key "context" อธิบายการปรับ
    """
    if not sig or sig.get("error"):
        return sig

    out  = deepcopy(sig)
    sign = _ml_sign(sig.get("direction", ""))

    # base confidence
    try:
        base = float(str(sig.get("confidence", "0")).replace("%", ""))
    except ValueError:
        base = 0.0

    fb = fund_bias(fund)
    sb = sent_bias(sent)
    wsum    = (W_FUND + W_SENT) or 1.0
    ctx_bias = (W_FUND * fb + W_SENT * sb) / wsum   # [-1,1] บวก=หนุนทอง

    reg_name = (regime or {}).get("regime", "UNKNOWN")
    reg_conf = (regime or {}).get("confidence", 0)

    new_conf  = base
    reasons   = []
    vetoed    = False
    agreement = 0.0

    # NEUTRAL ไม่ต้องปรับ (ไม่เทรดอยู่แล้ว)
    if sign == 0:
        out["context"] = {"regime": reg_name, "fund_bias": round(fb, 3),
                          "sent_bias": round(sb, 3), "ctx_bias": round(ctx_bias, 3),
                          "base_conf": base, "adj_conf": base, "delta": 0.0,
                          "vetoed": False, "reason": "NEUTRAL — ไม่ปรับ"}
        return out

    # 1) fundamental + sentiment (เห็นด้วย/ขัด ทิศ ML)
    agreement = sign * ctx_bias            # >0 หนุน, <0 ขัด
    delta_ctx = agreement * MAX_CTX
    new_conf += delta_ctx
    if abs(delta_ctx) >= 0.5:
        side = "หนุน" if delta_ctx > 0 else "ขัด"
        reasons.append(f"fund/sent {side} ({delta_ctx:+.1f})")

    # 2) regime
    reg_mult       = 1.0
    regime_opposes = False
    if reg_name in ("TRENDING_UP", "TRENDING_DOWN"):
        reg_sign = 1 if reg_name == "TRENDING_UP" else -1
        if reg_sign == sign:
            new_conf += REGIME_BONUS
            reasons.append(f"{reg_name} หนุน (+{REGIME_BONUS:.0f})")
        else:
            new_conf -= REGIME_PENALTY
            regime_opposes = True
            reasons.append(f"{reg_name} สวน ML (-{REGIME_PENALTY:.0f})")
    elif reg_name == "SIDEWAYS":
        reg_mult = SIDEWAYS_MULT
        reasons.append(f"SIDEWAYS damp x{SIDEWAYS_MULT}")
    elif reg_name == "VOLATILE":
        reg_mult = VOLATILE_MULT
        reasons.append(f"VOLATILE damp x{VOLATILE_MULT}")

    new_conf *= reg_mult
    new_conf  = _clip(new_conf, 0.0, 99.0)

    # 3) veto เมื่อบริบทขัดรุนแรง — fund/sent ขัดมาก หรือ trend สวนทิศ ML
    if (VETO and agreement <= VETO_AGREE) or (VETO_REGIME and regime_opposes):
        vetoed = True
        why = f"agreement {agreement:+.2f}" if agreement <= VETO_AGREE else f"{reg_name} สวนทิศ"
        reasons.append(f"VETO ({why})")
        out["direction"]  = "NEUTRAL —"
        out["confidence"] = f"{new_conf:.1f}%"
        out["entry"] = out["SL"] = out["TP"] = out["RR"] = None
    else:
        out["confidence"] = f"{new_conf:.1f}%"

    out["context"] = {
        "regime"    : reg_name,
        "regime_conf": reg_conf,
        "fund_bias" : round(fb, 3),
        "sent_bias" : round(sb, 3),
        "ctx_bias"  : round(ctx_bias, 3),
        "agreement" : round(agreement, 3),
        "base_conf" : round(base, 1),
        "adj_conf"  : round(new_conf, 1),
        "delta"     : round(new_conf - base, 1),
        "vetoed"    : vetoed,
        "reason"    : " · ".join(reasons) if reasons else "ไม่มีการปรับ",
    }
    return out


def adjust(sig: dict, df=None, tf: str = "M15") -> dict:
    """convenience: โหลด context เองแล้วปรับ (ถ้า SIGNAL_ADJUST=false → คืนเดิม)"""
    if not ENABLED:
        return sig
    ctx = load_context(df, tf)
    return adjust_signal(sig, regime=ctx["regime"], fund=ctx["fund"], sent=ctx["sent"])


# ── self-test: python gold_signal_adjuster.py ──────────────
if __name__ == "__main__":
    demo = {"direction": "UP ↑", "confidence": "72.0%",
            "entry": 2000.0, "SL": 1995.0, "TP": 2008.0, "RR": 1.6}

    print("base:", demo["confidence"])
    print("UP + bullish fund + trending_up:",
          adjust_signal(demo,
                        regime={"regime": "TRENDING_UP", "confidence": 80},
                        fund={"score": {"normalized": 60}},
                        sent={"sentiment": {"avg_net_score": 2}})["context"])
    print("UP + bearish fund + trending_down (ควร veto):",
          adjust_signal(demo,
                        regime={"regime": "TRENDING_DOWN", "confidence": 80},
                        fund={"score": {"normalized": -80}},
                        sent={"sentiment": {"avg_net_score": -3}})["context"])
    print("UP + sideways (damp):",
          adjust_signal(demo, regime={"regime": "SIDEWAYS", "confidence": 70},
                        fund={}, sent={})["context"])
