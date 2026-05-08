"""
=============================================================
 News Sentiment Analysis — Gold Trading
 ดึงข่าว + NLP วิเคราะห์ทิศทางทอง
=============================================================
 ติดตั้ง:
   pip install requests transformers torch newspaper3k
               beautifulsoup4 python-dotenv

 .env (optional — เพิ่ม coverage):
   NEWS_API_KEY=your_newsapi_key    # newsapi.org ฟรี 100 req/day
=============================================================
"""

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import requests
from dotenv import load_dotenv

load_dotenv()

DATA_DIR    = Path("./gold_data")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

# ════════════════════════════════════════════════════════════
#  KEYWORD DICTIONARIES
# ════════════════════════════════════════════════════════════

BULLISH_KEYWORDS = {
    # Macro bullish
    "inflation"         : 3, "cpi rose"        : 3, "inflation surged"  : 3,
    "rate cut"          : 3, "rate cuts"        : 3, "dovish"            : 2,
    "fed pivot"         : 3, "quantitative easing": 3, "stimulus"         : 2,
    "recession fears"   : 2, "recession risk"   : 2, "economic slowdown" : 2,
    "safe haven"        : 3, "flight to safety" : 3, "uncertainty"       : 1,
    "dollar weakened"   : 2, "dollar falls"     : 2, "dollar decline"    : 2,
    "geopolitical"      : 2, "war"              : 2, "conflict"          : 1,
    "sanctions"         : 1, "default risk"     : 2, "debt ceiling"      : 2,
    "banking crisis"    : 3, "bank failure"     : 3,
    # Gold specific
    "gold rallied"      : 3, "gold surged"      : 3, "gold buying"       : 2,
    "gold demand"       : 2, "central bank gold": 3, "gold reserves"     : 2,
    "gold backed"       : 2, "xau"              : 1,
}

BEARISH_KEYWORDS = {
    # Macro bearish
    "rate hike"         : 3, "hawkish"          : 2, "tightening"       : 2,
    "quantitative tightening": 2, "qt"           : 1,
    "strong dollar"     : 2, "dollar surged"    : 2, "dollar rally"     : 2,
    "risk on"           : 2, "risk appetite"    : 1, "equity rally"     : 1,
    "strong jobs"       : 2, "nonfarm payrolls beat": 2, "gdp beat"     : 1,
    "inflation eased"   : 3, "cpi fell"         : 3, "deflation"        : 2,
    "economic recovery" : 1,
    # Gold specific
    "gold fell"         : 3, "gold dropped"     : 3, "gold selling"     : 2,
    "gold outflows"     : 2, "etf outflows"     : 2,
}

# ════════════════════════════════════════════════════════════
#  NEWS SOURCES
# ════════════════════════════════════════════════════════════

RSS_FEEDS = [
    # Reuters
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.reuters.com/reuters/USDollarRoundup",
    # Kitco (gold specialist)
    "https://www.kitco.com/rss/kitco-news.xml",
    # FXStreet
    "https://www.fxstreet.com/rss/news",
    # MarketWatch
    "https://feeds.content.dowjones.io/public/rss/mw_topstories",
]

GOLD_KEYWORDS_FILTER = [
    "gold", "xau", "fed", "federal reserve", "dollar", "dxy",
    "inflation", "rate", "yield", "treasury", "fomc", "powell",
    "recession", "safe haven", "bullion", "precious metal",
    "geopolit", "ukraine", "middle east", "china",
]

def _fetch_rss(url: str, max_items: int = 10) -> list:
    try:
        from xml.etree import ElementTree as ET
        headers = {"User-Agent": "Mozilla/5.0 GoldBot/1.0"}
        r = requests.get(url, headers=headers, timeout=8)
        root = ET.fromstring(r.content)
        items = []
        for item in root.iter("item"):
            title = item.findtext("title", "") or ""
            desc  = item.findtext("description", "") or ""
            link  = item.findtext("link", "") or ""
            pub   = item.findtext("pubDate", "") or ""
            text  = f"{title} {desc}".lower()
            # กรองเฉพาะข่าวที่เกี่ยวกับทอง/macro
            if any(kw in text for kw in GOLD_KEYWORDS_FILTER):
                items.append({
                    "title"   : title.strip(),
                    "summary" : re.sub(r"<[^>]+>", "", desc)[:300].strip(),
                    "url"     : link,
                    "pub"     : pub,
                    "source"  : url.split("/")[2],
                })
            if len(items) >= max_items:
                break
        return items
    except Exception as e:
        print(f"[RSS] {url[:50]}... error: {e}")
        return []

def _fetch_newsapi(query: str = "gold price Federal Reserve", max: int = 20) -> list:
    if not NEWS_API_KEY:
        return []
    try:
        r = requests.get("https://newsapi.org/v2/everything", params={
            "q"         : query,
            "sortBy"    : "publishedAt",
            "language"  : "en",
            "pageSize"  : max,
            "apiKey"    : NEWS_API_KEY,
        }, timeout=8)
        articles = r.json().get("articles", [])
        return [{
            "title"  : a.get("title",""),
            "summary": a.get("description","")[:300] or "",
            "url"    : a.get("url",""),
            "pub"    : a.get("publishedAt",""),
            "source" : a.get("source",{}).get("name",""),
        } for a in articles if a.get("title")]
    except Exception as e:
        print(f"[NewsAPI] {e}")
        return []

def fetch_all_news() -> list:
    print("[NEWS] กำลังดึงข่าว...")
    articles = []
    for feed in RSS_FEEDS:
        articles += _fetch_rss(feed)
        time.sleep(0.3)

    # NewsAPI เพิ่มเติม
    if NEWS_API_KEY:
        articles += _fetch_newsapi("gold XAU Federal Reserve inflation")
        articles += _fetch_newsapi("dollar DXY treasury yield")

    # Deduplicate โดย title
    seen, unique = set(), []
    for a in articles:
        key = a["title"][:60].lower()
        if key not in seen:
            seen.add(key); unique.append(a)

    print(f"[NEWS] ดึงได้ {len(unique)} ข่าว")
    return unique


# ════════════════════════════════════════════════════════════
#  KEYWORD-BASED SENTIMENT (fast, no GPU needed)
# ════════════════════════════════════════════════════════════

def score_article_keywords(article: dict) -> dict:
    text = (article.get("title","") + " " +
            article.get("summary","")).lower()

    bull_score = sum(w for kw, w in BULLISH_KEYWORDS.items() if kw in text)
    bear_score = sum(w for kw, w in BEARISH_KEYWORDS.items() if kw in text)
    net        = bull_score - bear_score

    bull_found = [kw for kw in BULLISH_KEYWORDS if kw in text][:3]
    bear_found = [kw for kw in BEARISH_KEYWORDS if kw in text][:3]

    return {
        **article,
        "bull_score"   : bull_score,
        "bear_score"   : bear_score,
        "net_score"    : net,
        "sentiment"    : "bullish" if net > 1 else ("bearish" if net < -1 else "neutral"),
        "bull_keywords": bull_found,
        "bear_keywords": bear_found,
    }


# ════════════════════════════════════════════════════════════
#  TRANSFORMER NLP (ถ้ามี GPU หรือต้องการแม่นขึ้น)
# ════════════════════════════════════════════════════════════

_pipeline = None

def _load_nlp_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    try:
        from transformers import pipeline as hf_pipeline
        print("[NLP] โหลด FinancialBERT model...")
        _pipeline = hf_pipeline(
            "text-classification",
            model="ProsusAI/finbert",   # FinBERT trained on financial text
            tokenizer="ProsusAI/finbert",
            truncation=True,
            max_length=512,
        )
        print("[NLP] โหลดสำเร็จ")
    except Exception as e:
        print(f"[NLP] โหลด FinBERT ไม่ได้ ({e}) — ใช้ keyword แทน")
        _pipeline = None
    return _pipeline

def score_article_nlp(article: dict) -> dict:
    """ใช้ FinBERT วิเคราะห์ sentiment — แม่นกว่า keyword แต่ช้ากว่า"""
    pipe = _load_nlp_pipeline()
    if pipe is None:
        return score_article_keywords(article)

    text = f"{article.get('title','')}. {article.get('summary','')}".strip()[:512]
    try:
        result = pipe(text)[0]
        label  = result["label"].lower()   # positive/negative/neutral
        conf   = result["score"]

        # map finbert label → gold direction
        # positive news = bullish gold? Not always — depends on context
        # เราใช้ keyword ช่วยตัดสินใจ
        kw_result = score_article_keywords(article)
        net_kw    = kw_result["net_score"]

        if label == "positive" and net_kw > 0:
            sentiment = "bullish"
        elif label == "negative" and net_kw < 0:
            sentiment = "bearish"
        elif label == "positive" and net_kw < 0:
            # FinBERT บอก positive แต่ keyword บอก bearish → ดู confidence
            sentiment = "bullish" if conf > 0.8 else "neutral"
        elif label == "negative" and net_kw > 0:
            sentiment = "bearish" if conf > 0.8 else "neutral"
        else:
            sentiment = "neutral"

        return {
            **kw_result,
            "finbert_label": label,
            "finbert_conf" : round(conf, 3),
            "sentiment"    : sentiment,
        }
    except Exception as e:
        print(f"[NLP score] {e}")
        return score_article_keywords(article)


# ════════════════════════════════════════════════════════════
#  AGGREGATE SENTIMENT
# ════════════════════════════════════════════════════════════

def aggregate_sentiment(scored: list, use_nlp: bool = False) -> dict:
    if not scored:
        return {"bias": "NEUTRAL", "score": 0, "articles": 0}

    score_fn = score_article_nlp if use_nlp else score_article_keywords
    results  = [score_fn(a) for a in scored]

    bull = sum(1 for r in results if r["sentiment"] == "bullish")
    bear = sum(1 for r in results if r["sentiment"] == "bearish")
    neu  = sum(1 for r in results if r["sentiment"] == "neutral")
    total= len(results)

    net_scores  = [r["net_score"] for r in results]
    avg_net     = sum(net_scores) / total if total else 0

    # Top bullish/bearish articles
    top_bull = sorted([r for r in results if r["sentiment"] == "bullish"],
                      key=lambda x: x["net_score"], reverse=True)[:3]
    top_bear = sorted([r for r in results if r["sentiment"] == "bearish"],
                      key=lambda x: x["net_score"])[:3]

    bias = "BULLISH" if avg_net > 1 else ("BEARISH" if avg_net < -1 else "NEUTRAL")

    result = {
        "bias"          : bias,
        "avg_net_score" : round(avg_net, 2),
        "bull_count"    : bull,
        "bear_count"    : bear,
        "neutral_count" : neu,
        "total_articles": total,
        "bull_pct"      : round(bull/total*100, 1) if total else 0,
        "bear_pct"      : round(bear/total*100, 1) if total else 0,
        "top_bullish"   : [{"title": r["title"], "score": r["net_score"],
                            "keywords": r["bull_keywords"]} for r in top_bull],
        "top_bearish"   : [{"title": r["title"], "score": r["net_score"],
                            "keywords": r["bear_keywords"]} for r in top_bear],
        "updated"       : datetime.now(timezone.utc).isoformat(),
    }

    print(f"\n[SENTIMENT] {bias} | Bull:{bull} Bear:{bear} Neu:{neu} | avg={avg_net:+.2f}")
    for r in top_bull[:2]:
        print(f"  ↑ {r['title'][:80]}")
    for r in top_bear[:2]:
        print(f"  ↓ {r['title'][:80]}")

    return result


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════

def run_sentiment_analysis(use_nlp: bool = False) -> dict:
    articles  = fetch_all_news()
    sentiment = aggregate_sentiment(articles, use_nlp=use_nlp)

    # บันทึก
    out = {"sentiment": sentiment, "articles": articles[:50]}
    path = DATA_DIR / "sentiment.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f"💾 บันทึก sentiment ที่ {path}")
    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nlp", action="store_true", help="ใช้ FinBERT NLP (ช้ากว่า แต่แม่นกว่า)")
    args = parser.parse_args()
    run_sentiment_analysis(use_nlp=args.nlp)