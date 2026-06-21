# 🪙 Gold Trading System — สรุปโปรเจ็ค & แนวทางพัฒนาต่อ

ระบบเทรดทองคำ **XAUUSD** อัตโนมัติด้วย Machine Learning เชื่อมต่อ MetaTrader 5 (Exness)
ครบวงจร: ดึงข้อมูล → สร้างฟีเจอร์ → ทำนายด้วย ensemble model → แสดง dashboard → แจ้งเตือน → ส่งคำสั่งเทรดจริง

> อัปเดตล่าสุด: 21 มิ.ย. 2026

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
| [trades.html](trades.html) | หน้าเว็บดูการเข้าออเดอร์ของบอท (สรุป + ออเดอร์เปิดอยู่ + ประวัติ) |
| [gold_trader.py](gold_trader.py) | **บอทส่งคำสั่งเทรดจริง** + ส่ง Telegram order |
| [gold_telegram.py](gold_telegram.py) | แจ้งเตือน Telegram แยก 2 ช่อง (signal / order) |
| [gold_alert.py](gold_alert.py) | แจ้งเตือนผ่าน LINE Messaging API (ทางเลือก, ยังไม่ wire เข้า server) |

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
- [ ] **Risk management**: จำกัดขาดทุนต่อวัน (daily loss limit), max drawdown, หยุดบอทอัตโนมัติเมื่อถึงเพดาน
- [ ] **Position sizing ตามทุน**: คำนวณ lot จาก % ความเสี่ยงต่อไม้ + ระยะ SL แทน lot คงที่ 0.01
- [ ] **Market-hours guard**: เช็กความสดของ tick ก่อนส่งออเดอร์ — ข้ามตอนตลาดปิด เลี่ยง order ถูกปฏิเสธ (market closed)
- [ ] **หมุน/เพิกถอน secret**: bot token + รหัสเคยถูก commit ใน git history (first commit ก่อนทำ .gitignore) ควร revoke/เปลี่ยนเพื่อความปลอดภัย

### 🟡 B. การจัดการออเดอร์ (เพิ่มความสามารถ)
- [ ] **Exit logic ที่ดีขึ้น**: trailing stop / partial take-profit / เลื่อน SL ไป breakeven (ปัจจุบันพึ่ง SL-TP ตายตัวอย่างเดียว)
- [ ] **กันสัญญาณขัดกัน**: เมื่อหลาย TF ให้ทิศตรงข้าม ไม่ควรเปิดสวนซ้อนกัน — เพิ่ม logic ตัดสินใจรวม
- [ ] **ตรวจ spread/slippage** ก่อนเข้า — ข้ามถ้า spread กว้างผิดปกติ

### 🟢 C. การแจ้งเตือน / รายงาน
- [ ] ส่ง Telegram ตอน **OPEN_FAIL** (พร้อมเหตุผล retcode) เพื่อ debug ได้ทันที
- [ ] **สรุป P/L รายวัน/รายสัปดาห์** เข้า Telegram อัตโนมัติ
- [ ] Dashboard เพิ่ม: equity curve, win-rate, จำนวนไม้, สถิติย้อนหลังจาก trade logs

### 🔵 D. คุณภาพโมเดล / ผลิตภัณฑ์
- [ ] **Retrain schedule**: เทรนโมเดลใหม่อัตโนมัติเป็นรอบ + ใช้ [gold_walk_forward.py](gold_walk_forward.py) ตรวจ performance ก่อน deploy
- [ ] **รวม fundamental / sentiment / regime เข้าสัญญาณจริง** — มีโมดูลครบแล้วแต่ยังไม่ถูก wire เข้า ensemble เต็มที่
- [ ] เพิ่ม **README.md**, สคริปต์ start ทั้งระบบทีเดียว (server + bot), รันเป็น service/auto-start เมื่อเปิดเครื่อง
- [ ] **Backtest จริงจัง** ก่อนใช้เงินจริง — ยืนยัน edge ของกลยุทธ์บนข้อมูลย้อนหลังหลายช่วงตลาด

---

> ⚠️ **คำเตือน**: ระบบนี้เพื่อการศึกษา/วิจัย ปัจจุบันรันบนบัญชี **demo** เท่านั้น
> ก่อนนำไปใช้กับเงินจริง ต้องผ่านการ backtest + ทดสอบ demo เป็นระยะเวลานานพอ และมี risk management ครบถ้วน
