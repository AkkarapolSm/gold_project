"""
=============================================================
 Gold Risk Manager — Daily Loss Limit + Max Drawdown Guard
=============================================================
 ตรรกะล้วน (ไม่พึ่ง MetaTrader5) — รับแค่ balance / equity เป็นตัวเลข
 จึงทดสอบง่ายและ reuse ได้ทั้ง bot / backtest

 หน้าที่:
   - จำกัดขาดทุนต่อวัน (daily loss limit) เทียบ balance ต้นวัน
   - จำกัด max drawdown เทียบ peak equity (ระดับบัญชี)
   - "หยุดบอทอัตโนมัติ" (halt) เมื่อถึงเพดาน → ไม่เปิดออเดอร์ใหม่
   - reset เพดานขาดทุนรายวันอัตโนมัติเมื่อขึ้นวันใหม่
   - persist state ลงไฟล์ JSON เพื่อให้รอด restart

 หน่วยเป็น "เปอร์เซ็นต์":
   daily_loss_pct = 5.0  → ขาดทุนถึง 5% ของ balance ต้นวัน = หยุด
   max_dd_pct     = 10.0 → equity ต่ำกว่า peak เกิน 10% = หยุด
   ตั้งเป็น 0 = ปิดการเช็คเงื่อนไขนั้น
=============================================================
"""

import json
from datetime import date
from pathlib import Path


class RiskManager:
    """ตัวจัดการความเสี่ยงระดับบัญชี (daily loss + drawdown)"""

    def __init__(self, state_path, daily_loss_pct: float = 0.0,
                 max_dd_pct: float = 0.0):
        self.state_path     = Path(state_path)
        self.daily_loss_pct = float(daily_loss_pct or 0.0)
        self.max_dd_pct     = float(max_dd_pct or 0.0)

        # ── persisted state ──
        self.day               = None    # วันที่ (ISO) ของ baseline ปัจจุบัน
        self.day_start_balance = 0.0     # balance ตอนเริ่มวัน
        self.peak_equity       = 0.0     # equity สูงสุดที่เคยเห็น
        self.halted            = False   # หยุดเทรดอยู่ไหม
        self.halt_kind         = ""      # "DAILY_LOSS" | "MAX_DRAWDOWN"
        self.halt_reason       = ""
        self.alerted           = False   # แจ้งเตือน halt ไปแล้วหรือยัง (กันสแปม)

        self._load()

    @property
    def enabled(self) -> bool:
        """มีเงื่อนไขให้เช็คอย่างน้อย 1 อย่างไหม"""
        return self.daily_loss_pct > 0 or self.max_dd_pct > 0

    # ── persistence ──────────────────────────────────────
    def _load(self):
        if not self.state_path.exists():
            return
        try:
            d = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.day               = d.get("day")
            self.day_start_balance = float(d.get("day_start_balance", 0) or 0)
            self.peak_equity       = float(d.get("peak_equity", 0) or 0)
            self.halted            = bool(d.get("halted", False))
            self.halt_kind         = d.get("halt_kind", "") or ""
            self.halt_reason       = d.get("halt_reason", "") or ""
            self.alerted           = bool(d.get("alerted", False))
        except Exception:
            pass  # state เสียก็เริ่มใหม่ ไม่ critical

    def _save(self):
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps({
                "day"              : self.day,
                "day_start_balance": self.day_start_balance,
                "peak_equity"      : self.peak_equity,
                "halted"           : self.halted,
                "halt_kind"        : self.halt_kind,
                "halt_reason"      : self.halt_reason,
                "alerted"          : self.alerted,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def reset(self):
        """ล้าง state ทั้งหมด (ใช้ตอนสั่ง --reset-risk เพื่อปลด halt)"""
        self.day               = None
        self.day_start_balance = 0.0
        self.peak_equity       = 0.0
        self.halted            = False
        self.halt_kind         = ""
        self.halt_reason       = ""
        self.alerted           = False
        self._save()

    # ── core ─────────────────────────────────────────────
    def update(self, balance: float, equity: float) -> dict:
        """
        อัปเดต state + เช็คเพดานความเสี่ยง — เรียกทุกรอบ
        คืน snapshot dict (ดู key ตอนท้าย)
        """
        balance = float(balance)
        equity  = float(equity)
        today   = date.today().isoformat()

        # ── ขึ้นวันใหม่: รีเซ็ต baseline + ปลด halt ที่เป็นขาดทุนรายวัน ──
        if self.day != today:
            self.day               = today
            self.day_start_balance = balance
            if self.halt_kind == "DAILY_LOSS":
                self.halted      = False
                self.halt_kind   = ""
                self.halt_reason = ""
                self.alerted     = False

        # กันค่าเริ่มต้นเพี้ยน (เพิ่งสร้าง state)
        if self.day_start_balance <= 0:
            self.day_start_balance = balance
        if equity > self.peak_equity:
            self.peak_equity = equity

        # equity - balance ต้นวัน = (realized วันนี้) + (floating ตอนนี้)
        daily_pl     = equity - self.day_start_balance
        daily_pl_pct = (daily_pl / self.day_start_balance * 100) if self.day_start_balance else 0.0
        drawdown_pct = ((self.peak_equity - equity) / self.peak_equity * 100) if self.peak_equity > 0 else 0.0

        just_halted = False
        if not self.halted:
            if self.daily_loss_pct > 0 and daily_pl_pct <= -abs(self.daily_loss_pct):
                self.halted      = True
                self.halt_kind   = "DAILY_LOSS"
                self.halt_reason = (f"ขาดทุนรายวัน {daily_pl:+.2f} USD ({daily_pl_pct:+.2f}%) "
                                    f"ถึงเพดาน {self.daily_loss_pct:.2f}%")
                just_halted = True
            elif self.max_dd_pct > 0 and drawdown_pct >= abs(self.max_dd_pct):
                self.halted      = True
                self.halt_kind   = "MAX_DRAWDOWN"
                self.halt_reason = (f"Drawdown {drawdown_pct:.2f}% ถึงเพดาน {self.max_dd_pct:.2f}% "
                                    f"(peak equity {self.peak_equity:.2f})")
                just_halted = True

        self._save()

        return {
            "allowed"          : not self.halted,
            "halted"           : self.halted,
            "halt_kind"        : self.halt_kind,
            "halt_reason"      : self.halt_reason,
            "just_halted"      : just_halted,
            "daily_pl"         : daily_pl,
            "daily_pl_pct"     : daily_pl_pct,
            "drawdown_pct"     : drawdown_pct,
            "day_start_balance": self.day_start_balance,
            "peak_equity"      : self.peak_equity,
            "balance"          : balance,
            "equity"           : equity,
        }

    def mark_alerted(self):
        """ตั้งธงว่าแจ้งเตือน halt ไปแล้ว (กันส่งซ้ำทุกนาที)"""
        self.alerted = True
        self._save()


# ── self-test เล็กๆ: python gold_risk.py ───────────────────
if __name__ == "__main__":
    import tempfile, os
    p = Path(tempfile.gettempdir()) / "risk_state_test.json"
    if p.exists():
        p.unlink()

    rm = RiskManager(p, daily_loss_pct=5.0, max_dd_pct=10.0)
    print("enabled:", rm.enabled)

    r = rm.update(balance=500, equity=500)
    print("start      ->", r["allowed"], f"{r['daily_pl_pct']:+.2f}%")

    r = rm.update(balance=500, equity=490)        # -2%
    print("equity 490 ->", r["allowed"], f"{r['daily_pl_pct']:+.2f}%")

    r = rm.update(balance=500, equity=470)        # -6% => halt (daily loss)
    print("equity 470 ->", r["allowed"], "| halt:", r["halt_kind"], "| just:", r["just_halted"])

    r = rm.update(balance=500, equity=500)        # เด้งกลับ แต่ยัง halt อยู่ทั้งวัน
    print("recover    ->", r["allowed"], "| halt:", r["halt_kind"])

    p.unlink(missing_ok=True)
    print("OK")
