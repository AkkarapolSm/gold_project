# 🪙 Gold Trading System — สรุปโปรเจ็ค & แนวทางพัฒนาต่อ

ระบบเทรดทองคำ **XAUUSD** อัตโนมัติด้วย Machine Learning เชื่อมต่อ MetaTrader 5 (Exness)
ครบวงจร: ดึงข้อมูล → สร้างฟีเจอร์ → ทำนายด้วย ensemble model → แสดง dashboard → แจ้งเตือน → ส่งคำสั่งเทรดจริง

> อัปเดตล่าสุด: 22 มิ.ย. 2026 (เพิ่ม A. Risk · B. Order Mgmt · C. Alerts/Reporting · D. Model Quality)

---

## 1. สรุปโปรเจ็ค

### สถาปัตยกรรม / ไฟล์หลัก

| ไฟล์ | หน้าที่ |
|------|---------|
| [gold_mt5_pipeline.py](gold_mt5_pipeline.py) | เชื่อม MT5, ดึงแท่งเทียน OHLCV + tick, คำนวณ indicator, สัญญาณ rule-based |
| [gold_features.py](gold_features.py) | สร้าง feature สำหรับ ML |
| [gold_fundamental.py](gold_fundamental.py) | ดึงข้อมูลพื้นฐานเศรษฐกิจ (FRED API / yfinance) |
| [gold_sentiment.py](gold_sentiment.py) | วิเคราะห์ sentiment ข่าว (NewsAPI) |
| [gold_regime.py](gold_regime.py) | ตรวจจับสภาวะตลาด (trending / ranging) |
| [build_features.py](build_features.py) | สร้างไฟล์ feature CSV ทุก timeframe |
| [gold_model.py](gold_model.py) | เทรน/ทำนาย **Ensemble: XGBoost + LightGBM + LSTM** |
| [gold_walk_forward.py](gold_walk_forward.py) | ทดสอบโมเดลแบบ walk-forward optimization |
| [gold_server.py](gold_server.py) | **FastAPI backend** + background loop ทำนายทุก 1 นาที + ส่ง Telegram signal |
| [dashboard.html](dashboard.html) | หน้าเว็บแสดงสัญญาณ/indicator แบบ realtime |
| [trades.html](trades.html) | หน้าเว็บดูการเข้าออเดอร์ของบอท (สรุป + ออเดอร์เปิดอยู่ + ประวัติ + **equity curve / win-rate / profit factor**) |
| [gold_trader.py](gold_trader.py) | **บอทส่งคำสั่งเทรดจริง** + Telegram order + position sizing + risk/market guard + exit mgmt (trailing/BE/partial) + conflict & spread guard |
| [gold_risk.py](gold_risk.py) | **RiskManager**: daily loss limit + max drawdown + auto-halt (ตรรกะล้วน ไม่พึ่ง MT5) |
| [gold_telegram.py](gold_telegram.py) | แจ้งเตือน Telegram แยก 2 ช่อง (signal / order) |
| [gold_alert.py](gold_alert.py) | แจ้งเตือนผ่าน LINE Messaging API (ทางเลือก, ยังไม่ wire เข้า server) |
| [gold_signal_adjuster.py](gold_signal_adjuster.py) | **(D2)** รวม fundamental/sentiment/regime ปรับ confidence + veto |
| [gold_retrain.py](gold_retrain.py) | **(D1)** retrain orchestrator + WFO gate + backup/rollback (deploy ปลอดภัย) |
| [gold_backtest.py](gold_backtest.py) | **(D4)** backtest กลยุทธ์: equity curve / win-rate / profit factor / max drawdown |
| [README.md](README.md) · [start_all.ps1](start_all.ps1) | **(D3)** คู่มือ + สคริปต์เปิดทั้งระบบทีเดียว |

**โฟลเดอร์:** `gold_data/` (ข้อมูล+สัญญาณ+trade logs) · `gold_models/` (โมเดลที่เทรนแล้ว M15/M30/H1) · `gold_wfo/` (ผล walk-forward) · `venv/` (Python env)

### Data Flow

```
MT5 (XAUUSDm)
   │  ดึงแท่งเทียน 3 TF: M15 / M30 / H1
   ▼
build_features  →  Ensemble (XGB+LGBM+LSTM) predict_signal
   │
   ▼
gold_data/realtime_signals.json  ◄── gold_server เขียนทุก 1 นาที
   │
   ├──► dashboard.html (/)            แสดงสัญญาณ + indicator
   ├──► trades.html (/trades)         แสดงออเดอร์บอท + P/L
   ├──► Telegram "Gold_signals"       ถ้า conf ≥ 70%
   └──► gold_trader.py                ถ้า conf ≥ 80% → ส่ง order เข้า MT5
                                        └──► Telegram "Gold_Orders"
```

### Tech Stack
Python 3.14 · FastAPI / uvicorn · MetaTrader5 · pandas / numpy / ta / scipy · scikit-learn / xgboost / lightgbm / torch · requests / python-dotenv
(ดู [requirements.txt](requirements.txt))

---

## 2. สถานะปัจจุบัน (ใช้งานได้แล้ว ✅)

| ส่วน | สถานะ |
|------|-------|
| **gold_server** | รันพอร์ต **8001** → http://localhost:8001 (dashboard) + http://localhost:8001/trades |
| **gold_trader** | บอทเทรด loop ทุก 1 นาที — บัญชี **demo Exness-MT5Trial7**, balance **500 USD** |
| **MT5** | เชื่อมต่อสำเร็จ · Algo Trading **เปิดแล้ว** · symbol XAUUSDm เทรดได้เต็มรูปแบบ |
| **Telegram** | แยก 2 ช่อง — Gold_signals / Gold_Orders (ทดสอบส่งผ่านแล้ว) |
| **DevOps** | มี requirements.txt + .gitignore (กัน .env หลุดขึ้น git) |

### เกณฑ์การทำงาน (ปรับได้ใน [.env](.env))

| ขั้นตอน | เกณฑ์ | ตัวแปร |
|---------|-------|--------|
| ส่ง Signal → Telegram | conf **≥ 70%** (ไม่ใช่ NEUTRAL, กันส่งซ้ำ) | `TELEGRAM_SIGNAL_MIN_CONF` |
| บอทเข้าออเดอร์จริง | conf **≥ 80%** | `TRADE_MIN_CONF` |
| ส่ง Order → Telegram | ผูกกับการเข้าออเดอร์ (≥ 80%) | — |
| Lot size | 0.01 | `TRADE_LOT` |
| Max orders รวม | 5 ไม้ (pyramid ได้) | `TRADE_MAX_ORDERS` |
| Magic number | 20250323 | `TRADE_MAGIC` |

**ช่วง confidence:** `<70%` ไม่ส่งอะไร · `70–79%` แจ้ง signal แต่ยังไม่เข้าออเดอร์ · `≥80%` แจ้ง signal + เข้าออเดอร์ + แจ้ง order

### วิธีเปิดระบบ
```powershell
# เปิด MT5 (Exness) + login demo ค้างไว้ก่อน
.\venv\Scripts\Activate.ps1
uvicorn gold_server:app --port 8001      # dashboard + signal + telegram signal
python gold_trader.py                     # บอทเทรด + telegram order (อีกหน้าต่าง)
```

---

## 3. แนวทางพัฒนาต่อ

### 🔴 A. ความปลอดภัย / การจัดการความเสี่ยง (ควรทำก่อน)
- [x] **Risk management**: จำกัดขาดทุนต่อวัน (daily loss limit), max drawdown, หยุดบอทอัตโนมัติเมื่อถึงเพดาน — ทำใน [gold_risk.py](gold_risk.py) (`RiskManager`) + wire เข้า [gold_trader.py](gold_trader.py) (`check_risk`), persist สถานะที่ `gold_data/risk_state.json`, แจ้ง Telegram ตอน halt, ปลดด้วย `python gold_trader.py --reset-risk`
- [x] **Position sizing ตามทุน**: `calc_volume()` คิด lot จาก % ความเสี่ยงต่อไม้ + ระยะ SL (`TRADE_RISK_PCT`); ตั้ง `0` = ใช้ lot คงที่เหมือนเดิม
- [x] **Market-hours guard**: `is_market_open()` เช็ก `trade_mode` + ความสดของ tick (`TRADE_TICK_MAX_AGE`) ก่อนส่งออเดอร์ — ข้ามรอบตอนตลาดปิด เลี่ยง order ถูกปฏิเสธ
- [ ] **หมุน/เพิกถอน secret**: bot token + รหัสเคยถูก commit ใน git history (first commit ก่อนทำ .gitignore) ควร revoke/เปลี่ยนเพื่อความปลอดภัย ⚠️ *ต้องทำเองที่ผู้ให้บริการ (ดูขั้นตอนท้ายไฟล์)*

### 🟡 B. การจัดการออเดอร์ (เพิ่มความสามารถ)
- [x] **Exit logic ที่ดีขึ้น**: `manage_open_positions()` ทำ trailing stop + breakeven + partial TP (เป็นสัดส่วนของระยะ TP), แก้ SL เฉพาะตอน "แน่นขึ้น", เคารพ `trade_stops_level` ของโบรกเกอร์, persist สถานะที่ `gold_data/managed_positions.json` (กัน partial ซ้ำ) — toggle ผ่าน `TRADE_MANAGE_EXITS/BREAKEVEN/TRAILING/PARTIAL_TP`
- [x] **กันสัญญาณขัดกัน**: `resolve_conflict()` รวมหลาย TF เป็นทิศเดียว (เลือกฝั่ง conf รวมมากกว่า, สูสีภายใน `TRADE_CONFLICT_MARGIN` = งดเทรด) + **กันเปิดสวน**ของโพสิชันที่ถืออยู่
- [x] **ตรวจ spread** ก่อนเข้า: `open_order` ข้ามถ้า `ask-bid > TRADE_MAX_SPREAD` (กันเข้าในช่วงผันผวน/ข่าว) + ดัก slippage ด้วย `deviation`

### 🟢 C. การแจ้งเตือน / รายงาน
- [x] ส่ง Telegram ตอน **OPEN_FAIL** พร้อมเหตุผล (map `retcode` → ข้อความ เช่น market closed / no money / invalid stops) ทั้ง path `order_send=None` และ retcode ไม่ผ่าน — `_notify_open_fail()`
- [x] **สรุป P/L รายวัน/รายสัปดาห์** เข้า Telegram อัตโนมัติ: `report_pnl()` ดึง history deals (filter magic) คิด win-rate/profit factor/ดีสุด-แย่สุด · schedule ตาม `REPORT_DAILY_TIME`/`REPORT_WEEKLY_DAY` · สั่งเองได้ `python gold_trader.py --report daily|weekly`
- [x] Dashboard `/trades` เพิ่ม: **equity curve** (cumulative realized P/L, SVG ไม่ง้อ lib), **win-rate**, **profit factor**, จำนวนไม้ W/L, ดีสุด/แย่สุด — `gold_server.py` `_realized_stats()` (จาก MT5 history 30 วัน, group ตาม position_id)

### 🔵 D. คุณภาพโมเดล / ผลิตภัณฑ์
- [x] **Retrain schedule**: [gold_retrain.py](gold_retrain.py) — rebuild features → ประเมิน Walk-Forward (OOS) → ผ่าน gate `RETRAIN_GATE_DIR_ACC` เท่านั้นจึง deploy, retrain `EnsembleModel` (รูปแบบไฟล์เดียวกับ live) พร้อม **backup + rollback**, scheduler opt-in ผ่าน `RETRAIN_AUTO` · แก้บั๊ก [gold_walk_forward.py](gold_walk_forward.py) ที่ `retrain_latest` เคยเซฟทับ live models (ย้ายไป `gold_wfo/`)
- [x] **รวม fundamental / sentiment / regime เข้าสัญญาณจริง**: [gold_signal_adjuster.py](gold_signal_adjuster.py) เป็น context overlay ปรับ confidence + veto (ไม่ต้อง retrain, ไม่แตะ entry/SL/TP) wire เข้า background loop ของ [gold_server.py](gold_server.py) · เปิด/ปรับน้ำหนักผ่าน `SIGNAL_ADJUST` / `ADJUST_*`
- [x] เพิ่ม **[README.md](README.md)** ครบวงจร + **[start_all.ps1](start_all.ps1)** เปิด server+bot ทีเดียว + วิธี auto-start ด้วย Task Scheduler
- [x] **Backtest จริงจัง**: [gold_backtest.py](gold_backtest.py) จำลองกลยุทธ์จริง (SL/TP ATR, ถือทีละไม้, timeout) → equity curve, win-rate, **profit factor, max drawdown, expectancy** บันทึก `gold_wfo/backtest_*.{csv,json}` · หมายเหตุ in-sample → ใช้ walk-forward ดู edge OOS

> หมายเหตุการ deploy D: signal adjuster + backtest + retrain orchestrator ผ่านการ verify (compile/import/รันจริงกับ CSV) แล้ว · การ retrain/auto-retrain เต็มรูปแบบต้องมี MT5 + เวลาเทรน (โดยเฉพาะ LSTM) จึงควรรันช่วงตลาดปิด

---

> ⚠️ **คำเตือน**: ระบบนี้เพื่อการศึกษา/วิจัย ปัจจุบันรันบนบัญชี **demo** เท่านั้น
> ก่อนนำไปใช้กับเงินจริง ต้องผ่านการ backtest + ทดสอบ demo เป็นระยะเวลานานพอ และมี risk management ครบถ้วน

---

## ภาคผนวก A4 — หมุน/เพิกถอน secret (ต้องทำเองที่ผู้ให้บริการ)

secret เหล่านี้เคยถูก commit ลง git history ใน first commit (ก่อนเพิ่ม `.gitignore`) — แม้ตอนนี้ `.env` จะถูก ignore แล้ว แต่ค่าเดิม **ยังอยู่ใน history** ใครก็ตามที่เห็น repo จะดึงกลับมาได้ จึงควร revoke/เปลี่ยนใหม่:

| secret | วิธีเปลี่ยน |
|--------|-----------|
| `TELEGRAM_BOT_TOKEN` | คุยกับ **@BotFather** → `/revoke` (หรือ `/token`) เพื่อออก token ใหม่ แล้วอัปเดตใน `.env` |
| `MT5_PASSWORD` (บัญชี Real เดิมในคอมเมนต์ `.env`) | เปลี่ยนรหัสใน **Exness Personal Area** → Trading Accounts → Change password |
| `FRED_API_KEY` | ขอ key ใหม่ที่ fred.stlouisfed.org แล้วลบ key เก่า |
| `LINE_CHANNEL_TOKEN` | LINE Developers Console → channel → **Issue/Reissue** access token |

> หลังเปลี่ยน secret แล้ว ถ้าต้องการลบของเก่าออกจาก git history ด้วย ให้ใช้ `git filter-repo` หรือ **BFG Repo-Cleaner** (ระวัง: เขียน history ใหม่ ต้อง force-push และแจ้งคนที่ clone ไปแล้ว) — แต่ **การ revoke/เปลี่ยน secret คือสิ่งที่สำคัญที่สุด** เพราะตัดการใช้งานของค่าที่หลุดไปได้ทันที
