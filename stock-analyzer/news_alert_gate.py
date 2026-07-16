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
from oil_short_playbook import IsNonferrousGroup, IsOilShortGroup
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
_STRONG_NONFERROUS = (
    "铝价大涨", "铝价大跌", "电解铝", "氧化铝", "有色板块",
    "限电", "产能", "动力煤", "沪铝", "板块大涨",
)


# 盘外定点（含周末）：晚间/夜里扫隔夜资讯
_NEWS_WATCH_OFF_HOURS_SLOTS = (
    (18, 0),  # 18:00
    (22, 0),  # 22:00
)
_NEWS_WATCH_SLOT_TOLERANCE_MIN = 10


def IsInNewsWatchWindow(now: datetime | None = None) -> bool:
    """资讯哨兵窗口：工作日盘中连续区间 + 每天晚间定点 ±10 分钟。

    盘中（仅工作日）：09:30–11:30、13:00–15:00（由 cron */15 触发）。
    盘外（每天含周末）：18:00、22:00 ±10 分钟。
    """
    dt = now or datetime.now()
    minutes = dt.hour * 60 + dt.minute

    for hour, minute in _NEWS_WATCH_OFF_HOURS_SLOTS:
        slot = hour * 60 + minute
        if abs(minutes - slot) <= _NEWS_WATCH_SLOT_TOLERANCE_MIN:
            return True

    if dt.weekday() < 5:
        morning = 9 * 60 + 30 <= minutes <= 11 * 60 + 30
        afternoon = 13 * 60 <= minutes <= 15 * 60
        if morning or afternoon:
            return True
    return False


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


def BuildNewsAlertKey(group: str, title: str) -> str:
    """构建资讯告警去重键（按监控组 + 归一化标题）。"""
    day = datetime.now().strftime("%Y-%m-%d")
    group_id = (group or config.STOCK_GROUP or "default").strip() or "default"
    norm = NormalizeAlertTitle(title)
    digest = hashlib.sha1(norm.encode("utf-8")).hexdigest()[:12]
    return f"{day}:{group_id}:{digest}"


def IsNewsAlertClaimed(title: str, group: str | None = None) -> bool:
    """同监控组下该标题是否已认领（24h 内）。"""
    if not title.strip():
        return True
    group_id = (group or config.STOCK_GROUP or "default").strip() or "default"
    key = BuildNewsAlertKey(group_id, title)
    state = _PruneAlertDedup(_LoadAlertDedup())
    return key in state


def TryClaimNewsAlert(title: str, group: str | None = None) -> bool:
    """认领一条资讯告警；同监控组、同归一化标题 24 小时内不可重复。"""
    if not title.strip():
        return False
    group_id = (group or config.STOCK_GROUP or "default").strip() or "default"
    key = BuildNewsAlertKey(group_id, title)
    state = _PruneAlertDedup(_LoadAlertDedup())
    if key in state:
        logger.info(
            "资讯告警去重命中（群=%s），跳过: %s",
            group_id,
            title[:40],
        )
        return False
    state[key] = datetime.now().isoformat(timespec="seconds")
    _SaveAlertDedup(state)
    logger.info(
        "资讯告警已认领（群=%s）entries=%d: %s",
        group_id,
        len(state),
        title[:40],
    )
    return True


def TryClaimNewsAlertBundle(
    titles: list[str],
    group: str | None = None,
) -> bool:
    """批量认领相关标题；任一已认领则整批拒绝（同事件多标题变体）。"""
    cleaned = [t.strip() for t in titles if t and str(t).strip()]
    if not cleaned:
        return False
    group_id = (group or config.STOCK_GROUP or "default").strip() or "default"
    state = _PruneAlertDedup(_LoadAlertDedup())
    keys = [BuildNewsAlertKey(group_id, t) for t in cleaned]
    for key, title in zip(keys, cleaned):
        if key in state:
            logger.info(
                "资讯告警去重命中（群=%s 批量），跳过: %s",
                group_id,
                title[:40],
            )
            return False
    now_iso = datetime.now().isoformat(timespec="seconds")
    for key in keys:
        state[key] = now_iso
    _SaveAlertDedup(state)
    logger.info(
        "资讯告警已批量认领（群=%s）titles=%d entries=%d",
        group_id,
        len(keys),
        len(state),
    )
    return True


def CollectClaimTitles(
    news_bundle: dict[str, Any],
    headline: str,
) -> list[str]:
    """收集触发标题 + 新鲜催化/直接影响标题，用于同事件变体一并去重。"""
    titles: list[str] = []
    seen: set[str] = set()

    def _add(raw: str) -> None:
        text = str(raw or "").strip()
        if not text:
            return
        norm = NormalizeAlertTitle(text)
        if not norm or norm in seen:
            return
        seen.add(norm)
        titles.append(text)

    _add(headline)
    for item in _StrongCatalystHits(news_bundle):
        _add(str(item.get("title", "")))
    for item in _ImpactfulItems(news_bundle):
        _add(str(item.get("title", "")))
    return titles


def AlertDedupEntryCount() -> int:
    """当前本地去重条目数（运维日志用）。"""
    return len(_PruneAlertDedup(_LoadAlertDedup()))


def _FilterFresh(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [i for i in items if IsNewsFreshEnough(i)]


def _IsStrongCatalystItem(item: dict[str, Any]) -> bool:
    title = str(item.get("title", ""))
    content = str(item.get("content", ""))
    text = f"{title} {content}"
    is_cat = bool(item.get("is_catalyst")) or IsCatalystText(title, content)
    if not is_cat:
        return False
    if IsOilShortGroup():
        strong_words = _STRONG_OIL
    elif IsNonferrousGroup():
        strong_words = _STRONG_NONFERROUS
    else:
        strong_words = _STRONG_PHARMA
    if any(w in text for w in strong_words) or item.get("is_catalyst"):
        if IsOilShortGroup() or IsNonferrousGroup():
            return True
        if any(w in text for w in strong_words) or "授权" in text:
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
