"""资讯即时推送门控 — 规则优先，少耗 LLM。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import config
from oil_short_advisor import AggregateOilNewsAlert
from oil_short_playbook import IsOilShortGroup
from sector_catalyst_watch import IsCatalystText
from utils import SetupLogger

logger = SetupLogger(config.LOG_LEVEL)

_ALERT_DEDUP_PATH = Path(__file__).resolve().parent / ".news_alert_dedup.json"
_ALERT_TTL_HOURS = 4

_STRONG_PHARMA = (
    "海外授权", "授权重磅", "亿美元", "连板", "涨停潮",
    "板块大涨", "集体高开", "创新药板块", "医药指数",
)
_STRONG_OIL = (
    "原油大跌", "原油大涨", "布伦特", "地缘冲突", "制裁",
    "通航", "和谈", "停火", "霍尔木兹", "OPEC",
)


def IsInNewsWatchWindow(now: datetime | None = None) -> bool:
    """交易日资讯哨兵窗口：09:05-11:25、13:00-14:55（避开正式推送点）。"""
    dt = now or datetime.now()
    if dt.weekday() >= 5:
        return False
    minutes = dt.hour * 60 + dt.minute
    morning = 9 * 60 + 5 <= minutes <= 11 * 60 + 25
    afternoon = 13 * 60 <= minutes <= 14 * 60 + 55
    # 避开正式点附近
    near_scheduled = minutes in {
        9 * 60, 9 * 60 + 1, 9 * 60 + 2, 9 * 60 + 3, 9 * 60 + 4,
        11 * 60 + 26, 11 * 60 + 27, 11 * 60 + 28, 11 * 60 + 29, 11 * 60 + 30,
        11 * 60 + 31, 11 * 60 + 32, 11 * 60 + 33, 11 * 60 + 34,
        14 * 60 + 26, 14 * 60 + 27, 14 * 60 + 28, 14 * 60 + 29, 14 * 60 + 30,
        14 * 60 + 31, 14 * 60 + 32, 14 * 60 + 33, 14 * 60 + 34,
    }
    if near_scheduled:
        return False
    return morning or afternoon


def _LoadAlertDedup() -> dict[str, str]:
    if not _ALERT_DEDUP_PATH.exists():
        return {}
    try:
        raw = _ALERT_DEDUP_PATH.read_text(encoding="utf-8").strip()
        if not raw:
            return {}
        data = json.loads(raw)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        logger.warning("读取资讯告警去重失败: %s", exc)
    return {}


def _SaveAlertDedup(state: dict[str, str]) -> None:
    try:
        _ALERT_DEDUP_PATH.write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("写入资讯告警去重失败: %s", exc)


def _PruneAlertDedup(state: dict[str, str]) -> dict[str, str]:
    cutoff = datetime.now() - timedelta(hours=_ALERT_TTL_HOURS)
    kept: dict[str, str] = {}
    for key, ts_text in state.items():
        try:
            ts = datetime.fromisoformat(ts_text)
        except ValueError:
            continue
        if ts >= cutoff:
            kept[key] = ts_text
    return kept


def BuildNewsAlertKey(stock_code: str, title: str) -> str:
    """构建资讯告警去重键。"""
    day = datetime.now().strftime("%Y-%m-%d")
    digest = hashlib.sha1(title.strip().encode("utf-8")).hexdigest()[:12]
    return f"{day}:{stock_code.strip()}:{digest}"


def TryClaimNewsAlert(stock_code: str, title: str) -> bool:
    """认领一条资讯告警；同标题 4 小时内不可重复。"""
    if not title.strip():
        return False
    key = BuildNewsAlertKey(stock_code, title)
    state = _PruneAlertDedup(_LoadAlertDedup())
    if key in state:
        logger.info("资讯告警去重命中，跳过: %s", title[:40])
        return False
    state[key] = datetime.now().isoformat(timespec="seconds")
    _SaveAlertDedup(state)
    return True


def _ImpactfulItems(news_bundle: dict[str, Any]) -> list[dict[str, Any]]:
    from news_analyzer import GetImpactfulItems
    items = list(news_bundle.get("relevant_items") or news_bundle.get("scored_items") or [])
    if items:
        return GetImpactfulItems(items)
    return []


def _StrongCatalystHits(news_bundle: dict[str, Any]) -> list[dict[str, Any]]:
    """强催化剂命中列表（规则）。"""
    items = list(news_bundle.get("items") or news_bundle.get("relevant_items") or [])
    hits: list[dict[str, Any]] = []
    strong_words = _STRONG_OIL if IsOilShortGroup() else _STRONG_PHARMA
    for item in items:
        title = str(item.get("title", ""))
        content = str(item.get("content", ""))
        text = f"{title} {content}"
        is_cat = bool(item.get("is_catalyst")) or IsCatalystText(title, content)
        if not is_cat:
            continue
        if any(w in text for w in strong_words) or item.get("is_catalyst"):
            # 医药：需强词或同行+授权类；已在 IsCatalystText
            if IsOilShortGroup() or any(w in text for w in strong_words) or "授权" in text:
                hits.append(item)
    return hits


def EvaluateNewsAlert(
    news_bundle: dict[str, Any],
    news_score: float = 0.0,
) -> dict[str, Any]:
    """评估是否值得即时推送。返回 {worthy, reason, headline}。"""
    if IsOilShortGroup():
        alert = news_bundle.get("oil_news_alert") or AggregateOilNewsAlert(
            news_bundle, news_score,
        )
        level = str(alert.get("level", "none"))
        if level in ("strong_bull", "strong_bear"):
            return {
                "worthy": True,
                "reason": f"oil_{level}",
                "headline": str(alert.get("headline") or "")[:80],
            }
        oil_hits = _StrongCatalystHits(news_bundle)
        if oil_hits:
            title = str(oil_hits[0].get("title", ""))
            return {
                "worthy": True,
                "reason": "oil_catalyst",
                "headline": title[:80],
            }
        return {"worthy": False, "reason": "no_oil_alert", "headline": ""}

    impactful = _ImpactfulItems(news_bundle)
    if impactful:
        return {
            "worthy": True,
            "reason": "impactful_direct",
            "headline": str(impactful[0].get("title", ""))[:80],
        }
    hits = _StrongCatalystHits(news_bundle)
    if hits:
        return {
            "worthy": True,
            "reason": "pharma_catalyst",
            "headline": str(hits[0].get("title", ""))[:80],
        }
    return {"worthy": False, "reason": "no_pharma_alert", "headline": ""}


def IsNewsAlertWorthy(
    news_bundle: dict[str, Any],
    news_score: float = 0.0,
) -> bool:
    """是否触发即时推送。"""
    return bool(EvaluateNewsAlert(news_bundle, news_score).get("worthy"))
