# 🪙 Gold Trading System (XAUUSD)

ระบบเทรดทองคำ **XAUUSD** อัตโนมัติด้วย Machine Learning เชื่อมต่อ MetaTrader 5 (Exness)
ครบวงจร: ดึงข้อมูล → สร้างฟีเจอร์ → ทำนายด้วย ensemble (XGBoost + LightGBM + LSTM) →
ปรับด้วยบริบท (fundamental / sentiment / regime) → dashboard → แจ้งเตือน Telegram → ส่งคำสั่งเทรดจริง
พร้อม **risk management / order management / รายงาน / backtest / auto-retrain**

> ⚠️ **เพื่อการศึกษา/วิจัยเท่านั้น** — ปัจจุบันรันบนบัญชี **demo** ก่อนใช้เงินจริงต้อง backtest + ทดสอบ demo นานพอ
> รายละเอียดเชิงลึกของแต่ละส่วนดูที่ [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)

---

## เริ่มต้นเร็ว (Quick start)

```powershell
# 1. ติดตั้ง dependencies
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. ตั้งค่า .env (ดูหัวข้อ Configuration)

# 3. เปิด MetaTrader 5 (Exness) + login บัญชี demo ค้างไว้

# 4. เปิดทั้งระบบทีเดียว
.\start_all.ps1
#   → http://localhost:8001         dashboard สัญญาณ
#   → http://localhost:8001/trades  ออเดอร์บอท + equity curve / win-rate
```

เปิดแยกเองก็ได้:
```powershell
uvicorn gold_server:app --port 8001   # server + signal + telegram signal
python gold_trader.py                  # บอทเทรด (อีกหน้าต่าง)
```

---

## สถาปัตยกรรม

```
MT5 (XAUUSDm)  ── 3 TF: M15 / M30 / H1
   │
   ▼
build_features → Ensemble (XGB+LGBM+LSTM) → predict_signal
   │                                            │
   │                          (D2) signal_adjuster: fund/sent/regime ปรับ conf + veto
   ▼
gold_data/realtime_signals.json  ◄── gold_server เขียนทุก 1 นาที
   │
   ├──► dashboard.html (/)              สัญญาณ + indicator
   ├──► trades.html (/trades)           ออเดอร์บอท + equity curve + win-rate (C3)
   ├──► Telegram "Gold_signals"         ถ้า conf ≥ 70%
   └──► gold_trader.py                  ถ้า conf ≥ 80% → order เข้า MT5
                                          ├─ (A) risk guard / position sizing / market guard
                                          ├─ (B) trailing/breakeven/partial · conflict · spread
                                          ├─ (C1) OPEN_FAIL alert  (C2) สรุป P/L รายวัน/สัปดาห์
                                          └──► Telegram "Gold_Orders"
```

| ไฟล์ | หน้าที่ |
|------|---------|
| `gold_mt5_pipeline.py` | เชื่อม MT5, ดึงแท่งเทียน/tick, indicator |
| `gold_features.py` · `build_features.py` | สร้างฟีเจอร์ ML ทุก TF |
| `gold_model.py` | Ensemble XGB+LGBM+LSTM (train / predict) |
| `gold_fundamental.py` · `gold_sentiment.py` · `gold_regime.py` | บริบทตลาด (FRED/yfinance · ข่าว · regime) |
| `gold_signal_adjuster.py` | **(D2)** รวมบริบทเข้าสัญญาณจริง |
| `gold_server.py` | FastAPI backend + background predict loop |
| `dashboard.html` · `trades.html` | หน้าเว็บ realtime |
| `gold_trader.py` | บอทส่งคำสั่งเทรด + risk/order management + รายงาน |
| `gold_telegram.py` | แจ้งเตือน Telegram (signal / order) |
| `gold_walk_forward.py` | Walk-Forward Optimization (ประเมิน OOS) |
| `gold_retrain.py` | **(D1)** retrain + gate + deploy ปลอดภัย |
| `gold_backtest.py` | **(D4)** backtest กลยุทธ์ (equity/winrate/PF/maxDD) |
| `gold_risk.py` | **(A)** RiskManager (daily loss / drawdown) |

---

## Configuration (`.env`)

คัดมาเฉพาะที่สำคัญ — ดูคอมเมนต์ครบใน `.env`

**MT5 / API**
```
MT5_LOGIN=0            # 0 = เกาะ terminal ที่เปิด+login ค้างอยู่
MT5_SERVER=Exness-MT5Trial7
TELEGRAM_BOT_TOKEN=... TELEGRAM_SIGNAL_CHAT_ID=... TELEGRAM_ORDER_CHAT_ID=...
```

**A — Risk / Sizing / Market guard**
```
TRADE_RISK_PCT=0               # >0 = คิด lot จาก %เสี่ยง/ไม้ + SL ; 0 = ใช้ TRADE_LOT คงที่
TRADE_DAILY_LOSS_LIMIT_PCT=5   # หยุดเทรดเมื่อขาดทุนวันถึง % ของ balance ต้นวัน
TRADE_MAX_DRAWDOWN_PCT=10      # หยุดเมื่อ drawdown จาก peak equity ถึง %
TRADE_TICK_MAX_AGE=120         # tick เก่าเกิน N วิ = ตลาดปิด ข้ามรอบ
```

**B — Order management**
```
TRADE_MAX_SPREAD=0.50          # ข้ามถ้า spread > นี้ (USD)
TRADE_CONFLICT_GUARD=true      # รวมหลาย TF + กันเปิดสวน
TRADE_MANAGE_EXITS=true        # trailing + breakeven (+ partial ถ้าเปิด)
TRADE_TRAILING=true  TRADE_BREAKEVEN=true  TRADE_PARTIAL_TP=false
```

**C — Reporting**
```
REPORT_DAILY_TIME=23:59  REPORT_WEEKLY_DAY=sunday  REPORT_WEEKLY_TIME=23:55
```

**D — Signal adjuster / auto-retrain**
```
SIGNAL_ADJUST=true             # เปิด overlay fund/sent/regime
ADJUST_VETO=true               # veto เป็น NEUTRAL เมื่อบริบทขัดรุนแรง
RETRAIN_AUTO=false             # true = server ตั้ง retrain อัตโนมัติทุกสัปดาห์
RETRAIN_GATE_DIR_ACC=0.50      # ต้องผ่าน OOS dir_acc นี้จึง deploy
```

---

## คำสั่งที่ใช้บ่อย

```powershell
# บอทเทรด
python gold_trader.py                 # loop ทุก 1 นาที
python gold_trader.py --status        # ดูออเดอร์ + สถานะ risk
python gold_trader.py --closeall      # ปิดออเดอร์บอททั้งหมด
python gold_trader.py --reset-risk    # ปลด halt / รีเซ็ต baseline
python gold_trader.py --report daily  # ส่งสรุป P/L เข้า Telegram

# โมเดล / ประเมิน / backtest
python build_features.py              # สร้างฟีเจอร์ใหม่จาก MT5
python gold_model.py                  # train ensemble ทุก TF
python gold_walk_forward.py --tf M15  # ประเมิน out-of-sample
python gold_backtest.py --tf M15 --conf 80           # backtest กลยุทธ์
python gold_backtest.py --all --no-lstm              # ทุก TF (เร็ว)
python gold_retrain.py --dry-run --no-rebuild        # ประเมินก่อน deploy
python gold_retrain.py                               # retrain + deploy (ผ่าน gate)
```

---

## เปิดอัตโนมัติเมื่อเปิดเครื่อง (Windows Task Scheduler)

1. เปิด **Task Scheduler** → Create Task
2. **Triggers**: At log on (หรือ At startup)
3. **Actions**: Start a program
   - Program: `powershell.exe`
   - Arguments: `-ExecutionPolicy Bypass -File "C:\Users\akkar\gold_project\start_all.ps1"`
   - Start in: `C:\Users\akkar\gold_project`
4. ตั้ง MT5 ให้เปิด + login อัตโนมัติด้วย (MT5 จำ session ได้) ก่อน trigger

> หมายเหตุ: บอทต้องมี MT5 terminal เปิด+login อยู่ จึงควรให้ MT5 auto-start ก่อน start_all

---

## ⚠️ ความปลอดภัย

- รันบน **demo** เท่านั้นจนกว่าจะมั่นใจ — มี risk guard (A) แต่ไม่มีอะไรรับประกันกำไร
- `backtest` เป็น **in-sample** (โมเดลอาจเทรนทับช่วงนั้น) → ดู edge จริงจาก **walk-forward**
- **secret ที่เคยหลุดใน git history ควร revoke/เปลี่ยน** (Telegram token / MT5 password / FRED / LINE)
  ดูขั้นตอนที่ภาคผนวก A4 ใน [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)
- `.env` ถูก `.gitignore` แล้ว — อย่า commit ค่า secret
