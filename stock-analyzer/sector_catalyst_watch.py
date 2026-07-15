"""同行/板块催化剂监控 — 保留重磅板块新闻并打标。"""

from __future__ import annotations

import os
import re
from typing import Any

from oil_short_playbook import IsOilShortGroup

CATALYST_BONUS_LIMIT = 6

_DEFAULT_PHARMA_PEERS = (
    "迪哲医药",
    "哈药股份",
    "恒瑞医药",
    "百济神州",
    "信达生物",
    "药明康德",
    "凯莱英",
)

_PHARMA_CATALYST_PATTERN = re.compile(
    r"海外授权|授权交易|授权重磅|亿美元授权|license|BD交易|"
    r"连板|涨停潮|创新药板块|医药指数|多肽|减肥药|原料药|"
    r"创新药|板块普涨|集体高开",
    re.IGNORECASE,
)

_OIL_CATALYST_PATTERN = re.compile(
    r"原油|布伦特|WTI|OPEC|中东|伊朗|霍尔木兹|地缘|油服|"
    r"页岩油|制裁|冲突升级|通航|和谈|停火",
    re.IGNORECASE,
)

_DEFAULT_OIL_KEYWORDS = (
    "原油", "布伦特", "WTI", "OPEC", "中东", "伊朗", "海峡",
    "地缘", "油服", "页岩油", "油气", "石油",
)

_DEFAULT_PHARMA_SECTOR_EXTRA = (
    "授权", "BD", "海外授权", "创新药", "多肽", "减肥药",
    "原料药", "连板", "涨停潮", "医药指数",
)


def ResolvePeerWatchList() -> list[str]:
    """解析同行监控名单（env SECTOR_PEER_WATCH 可覆盖）。"""
    raw = os.getenv("SECTOR_PEER_WATCH", "").strip()
    if raw:
        return [p.strip() for p in raw.split(",") if p.strip()]
    if IsOilShortGroup():
        return []
    return list(_DEFAULT_PHARMA_PEERS)


def ExtraSectorKeywords() -> list[str]:
    """按组返回额外板块关键词。"""
    if IsOilShortGroup():
        return list(_DEFAULT_OIL_KEYWORDS)
    return list(_DEFAULT_PHARMA_SECTOR_EXTRA)


def IsCatalystText(title: str, content: str = "") -> bool:
    """标题/摘要是否命中催化剂。"""
    text = f"{title} {content}"
    if IsOilShortGroup():
        return bool(_OIL_CATALYST_PATTERN.search(text))
    peers = ResolvePeerWatchList()
    if any(p in text for p in peers):
        return True
    return bool(_PHARMA_CATALYST_PATTERN.search(text))


def TagCatalystItems(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """为条目打 is_catalyst 标记。"""
    tagged: list[dict[str, Any]] = []
    for item in items:
        copy = dict(item)
        title = str(copy.get("title", ""))
        content = str(copy.get("content", ""))
        if IsCatalystText(title, content):
            copy["is_catalyst"] = True
        tagged.append(copy)
    return tagged


def CollectCatalystBonus(
    global_df: Any,
    *,
    limit: int = CATALYST_BONUS_LIMIT,
) -> list[dict[str, str]]:
    """从全球快讯中额外捞取同行/催化剂条目（不依赖普通板块 head 截断）。"""
    if global_df is None or getattr(global_df, "empty", True):
        return []

    title_col = "标题" if "标题" in global_df.columns else global_df.columns[0]
    summary_col = "摘要" if "摘要" in global_df.columns else None
    time_col = "发布时间" if "发布时间" in global_df.columns else ""
    link_col = "链接" if "链接" in global_df.columns else ""

    peers = ResolvePeerWatchList()
    items: list[dict[str, str]] = []
    for _, row in global_df.iterrows():
        if len(items) >= limit:
            break
        title = str(row.get(title_col, "")).strip()
        if not title:
            continue
        summary = str(row.get(summary_col, "")) if summary_col else ""
        if not IsCatalystText(title, summary) and not any(p in title or p in summary for p in peers):
            continue
        items.append({
            "title": title,
            "content": summary[:300] + ("..." if len(summary) > 300 else ""),
            "time": str(row.get(time_col, "")).strip(),
            "source": "东方财富",
            "category": "sector",
            "url": str(row.get(link_col, "")).strip(),
            "is_catalyst": True,
        })
    return items


def MergeSectorWithCatalystBonus(
    sector_items: list[dict[str, Any]],
    catalyst_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """合并板块新闻与催化剂加成，标题去重，催化剂优先。"""
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _Add(item: dict[str, Any]) -> None:
        key = str(item.get("title", "")).strip().lower()
        if not key or key in seen:
            return
        seen.add(key)
        copy = dict(item)
        if copy.get("is_catalyst") in (True, "true", "1"):
            copy["is_catalyst"] = True
        elif IsCatalystText(str(copy.get("title", "")), str(copy.get("content", ""))):
            copy["is_catalyst"] = True
        merged.append(copy)

    for item in catalyst_items:
        _Add(item)
    for item in sector_items:
        _Add(item)
    return merged
