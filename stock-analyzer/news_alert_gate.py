"""资讯即时推送门控 — 规则优先，少耗 LLM；含新鲜度与去重。"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import config
from oil_short_advisor import AggregateOilNewsAlert
from oil_short_playbook import IsOilShortGroup
from sector_catalyst_watch import IsCatalystText
from utils import SetupLogger

logger = SetupLogger(config.LOG_LEVEL)

_ALERT_DEDUP_PATH = Path(__file__).resolve().parent / ".news_alert_dedup.json"
_ALERT_TTL_HOURS = 24
_NEWS_MAX_AGE_HOURS = 18

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


def ParseNewsItemDateTime(item: dict[str, Any]) -> datetime | None:
    """解析资讯发布时间；仅时分秒则视为当日该时刻。"""
    raw = str(
        item.get("time") or item.get("pub_time") or item.get("发布时间") or ""
    ).strip()
    if not raw:
        return None
    now = datetime.now()
    m = re.fullmatch(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", raw)
    if m:
        hour, minute = int(m.group(1)), int(m.group(2))
        second = int(m.group(3) or 0)
        try:
            return now.replace(hour=hour, minute=minute, second=second, microsecond=0)
        except ValueError:
            return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw[:19], fmt)
        except ValueError:
            continue
    digits = re.sub(r"\D", "", raw)
    if len(digits) >= 12:
        try:
            return datetime.strptime(digits[:14].ljust(14, "0")[:14], "%Y%m%d%H%M%S")
        except ValueError:
            pass
    if len(digits) >= 8:
        try:
            d = date(int(digits[:4]), int(digits[4:6]), int(digits[6:8]))
            return datetime(d.year, d.month, d.day, 0, 0, 0)
        except ValueError:
            return None
    return None


def IsNewsFreshEnough(
    item: dict[str, Any],
    max_age_hours: float = _NEWS_MAX_AGE_HOURS,
    now: datetime | None = None,
) -> bool:
    """发布时间须在近 max_age_hours 内；无有效时间视为不新鲜。"""
    parsed = ParseNewsItemDateTime(item)
    if parsed is None:
        return False
    ref = now or datetime.now()
    age = ref - parsed
    if age < timedelta(0):
        return age > timedelta(hours=-1)
    ok = age <= timedelta(hours=max_age_hours)
    if not ok:
        title = str(item.get("title", ""))[:40]
        hours = age.total_seconds() / 3600.0
        logger.info("news-watch: skip stale — %s age=%.1fh", title, hours)
    return ok


def NormalizeAlertTitle(title: str) -> str:
    """归一化标题用于去重（去标点/空白/数字）。"""
    text = re.sub(
        r"[（()）\s\"\"''\"\"、，。：:・·\-—_【】\[\]！!？?《》<>…·]+",
        "",
        title,
    )
    text = re.sub(r"\d+\.?\d*", "N", text)
    return text[:48]


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
    """构建资讯告警去重键（按归一化标题）。"""
    day = datetime.now().strftime("%Y-%m-%d")
    norm = NormalizeAlertTitle(title)
    digest = hashlib.sha1(norm.encode("utf-8")).hexdigest()[:12]
    return f"{day}:{stock_code.strip()}:{digest}"


def TryClaimNewsAlert(stock_code: str, title: str) -> bool:
    """认领一条资讯告警；同归一化标题 24 小时内不可重复。"""
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


def _FilterFresh(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [i for i in items if IsNewsFreshEnough(i)]


def _IsStrongCatalystItem(item: dict[str, Any]) -> bool:
    title = str(item.get("title", ""))
    content = str(item.get("content", ""))
    text = f"{title} {content}"
    is_cat = bool(item.get("is_catalyst")) or IsCatalystText(title, content)
    if not is_cat:
        return False
    strong_words = _STRONG_OIL if IsOilShortGroup() else _STRONG_PHARMA
    if any(w in text for w in strong_words) or item.get("is_catalyst"):
        if IsOilShortGroup() or any(w in text for w in strong_words) or "授权" in text:
            return True
    return False


def _ImpactfulItems(news_bundle: dict[str, Any]) -> list[dict[str, Any]]:
    from news_analyzer import GetImpactfulItems
    items = list(news_bundle.get("relevant_items") or news_bundle.get("scored_items") or [])
    if not items:
        return []
    return _FilterFresh(GetImpactfulItems(items))


def _StrongCatalystHits(news_bundle: dict[str, Any]) -> list[dict[str, Any]]:
    """强催化剂命中列表（仅新鲜）。"""
    items = list(news_bundle.get("items") or news_bundle.get("relevant_items") or [])
    return [i for i in items if _IsStrongCatalystItem(i) and IsNewsFreshEnough(i)]


def _HasAnyStrongCatalyst(news_bundle: dict[str, Any]) -> bool:
    items = list(news_bundle.get("items") or news_bundle.get("relevant_items") or [])
    return any(_IsStrongCatalystItem(i) for i in items)


def _FindItemByHeadline(
    news_bundle: dict[str, Any],
    headline: str,
) -> dict[str, Any] | None:
    head = headline.strip()
    if not head:
        return None
    for pool in (
        list(news_bundle.get("relevant_items") or []),
        list(news_bundle.get("items") or []),
        list(news_bundle.get("scored_items") or []),
    ):
        for item in pool:
            title = str(item.get("title", ""))
            if head in title or title in head:
                return item
    return None


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
            headline = str(alert.get("headline") or "")[:80]
            matched = _FindItemByHeadline(news_bundle, headline)
            if matched is not None:
                if not IsNewsFreshEnough(matched):
                    return {"worthy": False, "reason": "stale_news", "headline": ""}
            else:
                fresh_impact = _FilterFresh([
                    i for i in (list(news_bundle.get("relevant_items") or []))
                    if str(i.get("impact", "")) in ("涨", "跌")
                ])
                if not fresh_impact:
                    return {
                        "worthy": False,
                        "reason": "no_fresh_catalyst",
                        "headline": "",
                    }
                headline = str(fresh_impact[0].get("title", "") or headline)[:80]
            return {
                "worthy": True,
                "reason": f"oil_{level}",
                "headline": headline,
            }
        oil_hits = _StrongCatalystHits(news_bundle)
        if oil_hits:
            return {
                "worthy": True,
                "reason": "oil_catalyst",
                "headline": str(oil_hits[0].get("title", ""))[:80],
            }
        if _HasAnyStrongCatalyst(news_bundle):
            return {"worthy": False, "reason": "no_fresh_catalyst", "headline": ""}
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

    from news_analyzer import GetImpactfulItems
    all_rel = list(news_bundle.get("relevant_items") or news_bundle.get("scored_items") or [])
    raw_impactful = GetImpactfulItems(all_rel) if all_rel else []
    if raw_impactful and not _FilterFresh(raw_impactful):
        return {"worthy": False, "reason": "stale_news", "headline": ""}
    if _HasAnyStrongCatalyst(news_bundle):
        return {"worthy": False, "reason": "no_fresh_catalyst", "headline": ""}
    return {"worthy": False, "reason": "no_pharma_alert", "headline": ""}


def IsNewsAlertWorthy(
    news_bundle: dict[str, Any],
    news_score: float = 0.0,
) -> bool:
    """是否触发即时推送。"""
    return bool(EvaluateNewsAlert(news_bundle, news_score).get("worthy"))
