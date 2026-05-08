"""
=============================================================
 Fundamental Data Collector
 DXY · US10Y Yield · CPI · Fed Meeting Schedule
=============================================================
 ติดตั้ง:
   pip install requests pandas yfinance fredapi python-dotenv

 .env:
   FRED_API_KEY=your_fred_api_key   # ฟรีที่ fred.stlouisfed.org
=============================================================
"""

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path("./gold_data")
DATA_DIR.mkdir(exist_ok=True)

FRED_KEY = os.getenv("FRED_API_KEY", "")

# ════════════════════════════════════════════════════════════
#  1. DXY — Dollar Index (realtime ผ่าน yfinance)
# ════════════════════════════════════════════════════════════

def fetch_dxy(period="5d", interval="15m") -> dict:
    """
    ดึง DXY (Dollar Index) realtime
    DXY ขึ้น → ทองมักลง | DXY ลง → ทองมักขึ้น
    """
    try:
        dxy = yf.Ticker("DX-Y.NYB")
        df  = dxy.history(period=period, interval=interval)
        if df.empty:
            raise ValueError("DXY data empty")

        last     = df.iloc[-1]
        prev     = df.iloc[-2]
        change   = last["Close"] - prev["Close"]
        change_p = change / prev["Close"] * 100

        # trend 1D
        df_1d = dxy.history(period="30d", interval="1d")
        ma20  = df_1d["Close"].rolling(20).mean().iloc[-1]
        trend = "bullish" if last["Close"] > ma20 else "bearish"

        result = {
            "value"      : round(last["Close"], 3),
            "change"     : round(change, 3),
            "change_pct" : round(change_p, 3),
            "high_5d"    : round(df["High"].max(), 3),
            "low_5d"     : round(df["Low"].min(), 3),
            "ma20"       : round(ma20, 3),
            "trend"      : trend,
            "gold_bias"  : "bearish" if trend == "bullish" else "bullish",
            "updated"    : datetime.now(timezone.utc).isoformat(),
        }
        print(f"[DXY] {result['value']} ({'+' if change>=0 else ''}{change_p:.2f}%) trend={trend}")
        return result
    except Exception as e:
        print(f"[DXY ERROR] {e}")
        return {}


# ════════════════════════════════════════════════════════════
#  2. US10Y Yield (realtime)
# ════════════════════════════════════════════════════════════

def fetch_us10y(period="5d", interval="15m") -> dict:
    """
    US 10-Year Treasury Yield
    Yield สูงขึ้น → ทองมักแรงกดดัน (bond แข่งขัน)
    Yield ต่ำลง  → ทองมักได้รับความสนใจมากขึ้น
    """
    try:
        tnx = yf.Ticker("^TNX")
        df  = tnx.history(period=period, interval=interval)
        if df.empty:
            raise ValueError("US10Y empty")

        last   = df.iloc[-1]
        prev   = df.iloc[-2]
        change = last["Close"] - prev["Close"]

        df_1d  = tnx.history(period="30d", interval="1d")
        ma20   = df_1d["Close"].rolling(20).mean().iloc[-1]
        trend  = "rising" if last["Close"] > ma20 else "falling"

        result = {
            "value"     : round(last["Close"], 4),
            "change"    : round(change, 4),
            "ma20"      : round(ma20, 4),
            "trend"     : trend,
            "gold_bias" : "bearish" if trend == "rising" else "bullish",
            "updated"   : datetime.now(timezone.utc).isoformat(),
        }
        print(f"[US10Y] {result['value']}% trend={trend}")
        return result
    except Exception as e:
        print(f"[US10Y ERROR] {e}")
        return {}


# ════════════════════════════════════════════════════════════
#  3. CPI + Inflation Data (FRED API — ฟรี)
# ════════════════════════════════════════════════════════════

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

def _fred_get(series_id: str, limit: int = 12) -> list:
    if not FRED_KEY:
        return []
    try:
        r = requests.get(FRED_BASE, params={
            "series_id"      : series_id,
            "api_key"        : FRED_KEY,
            "file_type"      : "json",
            "sort_order"     : "desc",
            "limit"          : limit,
        }, timeout=10)
        data = r.json()
        obs  = [o for o in data.get("observations", []) if o["value"] != "."]
        return obs
    except Exception as e:
        print(f"[FRED {series_id}] {e}")
        return []

def fetch_cpi() -> dict:
    """
    CPI YoY — เงินเฟ้อสูง → ทองมักเป็น hedge
    Series: CPIAUCSL (US CPI All Urban)
    """
    obs = _fred_get("CPIAUCSL", 24)
    if len(obs) < 13:
        return {}

    latest   = float(obs[0]["value"])
    prev_yr  = float(obs[12]["value"])
    yoy      = round((latest - prev_yr) / prev_yr * 100, 2)
    mom      = round((latest - float(obs[1]["value"])) / float(obs[1]["value"]) * 100, 2)

    result = {
        "latest_index" : round(latest, 2),
        "yoy_pct"      : yoy,
        "mom_pct"      : mom,
        "date"         : obs[0]["date"],
        "trend"        : "rising" if yoy > float(obs[1]["value"]) else "falling",
        "gold_bias"    : "bullish" if yoy > 2.5 else "neutral",
        "note"         : f"CPI YoY={yoy}% {'↑ inflation hedge' if yoy>2.5 else '↓ less pressure'}",
        "updated"      : datetime.now(timezone.utc).isoformat(),
    }
    print(f"[CPI] YoY={yoy}% MoM={mom}%")
    return result

def fetch_pce() -> dict:
    """PCE Price Index — Fed ชอบดูตัวนี้มากกว่า CPI"""
    obs = _fred_get("PCEPI", 24)
    if len(obs) < 13:
        return {}
    latest  = float(obs[0]["value"])
    prev_yr = float(obs[12]["value"])
    yoy     = round((latest - prev_yr) / prev_yr * 100, 2)
    return {
        "yoy_pct"   : yoy,
        "date"      : obs[0]["date"],
        "gold_bias" : "bullish" if yoy > 2.0 else "neutral",
        "updated"   : datetime.now(timezone.utc).isoformat(),
    }

def fetch_real_yield() -> dict:
    """
    TIPS 10Y Real Yield (^TNX - CPI approx)
    Real yield ติดลบ → ทองแข็งแกร่ง
    """
    obs = _fred_get("DFII10", 5)   # 10Y TIPS yield from FRED
    if not obs:
        return {}
    latest = float(obs[0]["value"])
    return {
        "value"    : latest,
        "date"     : obs[0]["date"],
        "gold_bias": "bullish" if latest < 0 else ("neutral" if latest < 1.5 else "bearish"),
        "note"     : "Real yield ติดลบ = ทองได้เปรียบ" if latest < 0 else f"Real yield {latest}%",
        "updated"  : datetime.now(timezone.utc).isoformat(),
    }


# ════════════════════════════════════════════════════════════
#  4. Fed Meeting Schedule + Dot Plot
# ════════════════════════════════════════════════════════════

# วันประชุม FOMC 2025-2026 (อัปเดตได้จาก federalreserve.gov)
FOMC_DATES_2025_2026 = [
    "2025-01-29", "2025-03-19", "2025-05-07",
    "2025-06-18", "2025-07-30", "2025-09-17",
    "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29",
    "2026-06-17", "2026-07-29", "2026-09-16",
    "2026-10-28", "2026-12-09",
]

def fetch_fed_schedule() -> dict:
    """
    วันประชุม FOMC + จำนวนวันถึงประชุมครั้งถัดไป
    ตลาดมักผันผวนสูงในช่วง ±3 วัน ก่อน/หลังประชุม
    """
    today  = datetime.now(timezone.utc).date()
    dates  = [datetime.strptime(d, "%Y-%m-%d").date() for d in FOMC_DATES_2025_2026]
    future = sorted([d for d in dates if d >= today])

    if not future:
        return {"note": "ไม่มีข้อมูล FOMC ปีนี้"}

    next_date  = future[0]
    days_until = (next_date - today).days

    # ดึง Fed Funds Rate ปัจจุบัน
    fed_rate_obs = _fred_get("FEDFUNDS", 3)
    current_rate = float(fed_rate_obs[0]["value"]) if fed_rate_obs else None

    result = {
        "next_meeting"  : str(next_date),
        "days_until"    : days_until,
        "upcoming"      : [str(d) for d in future[:4]],
        "current_rate"  : current_rate,
        "high_volatility": days_until <= 3,
        "gold_note"     : (
            "⚠️ ใกล้ประชุม Fed — ความผันผวนสูง ระวัง SL ถูกกิน"
            if days_until <= 3
            else f"ประชุม Fed อีก {days_until} วัน"
        ),
        "updated"       : datetime.now(timezone.utc).isoformat(),
    }
    print(f"[FED] ประชุมครั้งถัดไป: {next_date} (อีก {days_until} วัน)")
    return result

def fetch_fed_rate_expectations() -> dict:
    """
    ดึง Fed Funds Futures implied rate จาก FRED
    ใช้ประเมิน market expectation ว่า Fed จะขึ้น/ลง/คง rate
    """
    obs = _fred_get("FEDFUNDS", 12)
    if len(obs) < 2:
        return {}

    rates  = [float(o["value"]) for o in obs[:6]]
    latest = rates[0]
    trend  = "cutting" if rates[0] < rates[2] else ("hiking" if rates[0] > rates[2] else "holding")

    return {
        "current_rate"   : latest,
        "trend"          : trend,
        "gold_bias"      : "bullish" if trend == "cutting" else ("bearish" if trend == "hiking" else "neutral"),
        "note"           : f"Fed rate trend: {trend} ({latest}%)",
        "updated"        : datetime.now(timezone.utc).isoformat(),
    }


# ════════════════════════════════════════════════════════════
#  5. คำนวณ Fundamental Score รวม
# ════════════════════════════════════════════════════════════

def compute_fundamental_score(fund: dict) -> dict:
    """
    รวม fundamental signals ออกมาเป็น score เดียว
    +1 = bullish gold, -1 = bearish gold, 0 = neutral
    """
    score   = 0
    signals = []

    # DXY
    dxy = fund.get("dxy", {})
    if dxy.get("gold_bias") == "bullish":
        score += 2; signals.append("DXY ลง → ทองได้แรงหนุน")
    elif dxy.get("gold_bias") == "bearish":
        score -= 2; signals.append("DXY ขึ้น → ทองถูกกดดัน")

    # US10Y
    y10 = fund.get("us10y", {})
    if y10.get("gold_bias") == "bullish":
        score += 2; signals.append("Yield ลง → ทองได้เปรียบ")
    elif y10.get("gold_bias") == "bearish":
        score -= 2; signals.append("Yield ขึ้น → ทองถูกกดดัน")

    # Real Yield
    ry = fund.get("real_yield", {})
    if ry.get("gold_bias") == "bullish":
        score += 2; signals.append("Real yield ติดลบ → ทองแข็งแกร่ง")
    elif ry.get("gold_bias") == "bearish":
        score -= 1; signals.append("Real yield สูง → ทองแรงกดดัน")

    # CPI
    cpi = fund.get("cpi", {})
    if cpi.get("gold_bias") == "bullish":
        score += 1; signals.append(f"CPI สูง → inflation hedge")

    # Fed Rate
    fed_rate = fund.get("fed_rate", {})
    if fed_rate.get("gold_bias") == "bullish":
        score += 1; signals.append("Fed กำลังลด rate → bullish gold")
    elif fed_rate.get("gold_bias") == "bearish":
        score -= 1; signals.append("Fed กำลังขึ้น rate → bearish gold")

    # FOMC proximity
    fomc = fund.get("fomc", {})
    if fomc.get("high_volatility"):
        signals.append("⚠️ ใกล้ประชุม Fed — ระวังความผันผวน")

    max_score = 8
    normalized = round(score / max_score * 100, 1)

    return {
        "raw_score"  : score,
        "normalized" : normalized,
        "bias"       : "BULLISH" if score > 2 else ("BEARISH" if score < -2 else "NEUTRAL"),
        "signals"    : signals,
        "updated"    : datetime.now(timezone.utc).isoformat(),
    }


# ════════════════════════════════════════════════════════════
#  6. MAIN — ดึงและบันทึกทุกอย่าง
# ════════════════════════════════════════════════════════════

def fetch_all_fundamentals() -> dict:
    print("\n[FUNDAMENTAL] กำลังดึงข้อมูล...")
    fund = {
        "dxy"        : fetch_dxy(),
        "us10y"      : fetch_us10y(),
        "cpi"        : fetch_cpi(),
        "pce"        : fetch_pce(),
        "real_yield" : fetch_real_yield(),
        "fomc"       : fetch_fed_schedule(),
        "fed_rate"   : fetch_fed_rate_expectations(),
    }
    fund["score"] = compute_fundamental_score(fund)

    # บันทึก
    out_path = DATA_DIR / "fundamentals.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fund, f, ensure_ascii=False, indent=2, default=str)

    score = fund["score"]
    print(f"\n[FUNDAMENTAL SCORE] {score['normalized']:+.1f}% → {score['bias']}")
    for s in score["signals"]:
        print(f"  • {s}")

    return fund


if __name__ == "__main__":
    data = fetch_all_fundamentals()
    print(f"\n💾 บันทึกที่ gold_data/fundamentals.json")