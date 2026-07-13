"""资讯评分上下文 — 公司主业、板块题材重叠度、盘面背离。"""

from __future__ import annotations

from typing import Any

import config
from utils import SafeFloat, SetupLogger

logger = SetupLogger(config.LOG_LEVEL)

_OVERLAP_HIGH_THRESHOLD = 2
_OVERLAP_MED_THRESHOLD = 1


def _ExtractConceptTopNames(data: dict[str, Any] | None, limit: int = 5) -> list[str]:
    """提取概念板块 TOP 名称列表。"""
    if not data:
        return []
    concept = data.get("concept")
    if concept is None or getattr(concept, "empty", True):
        return []
    names: list[str] = []
    for i in range(min(limit, len(concept))):
        name = str(concept.iloc[i].get("板块名称", "")).strip()
        if name:
            names.append(name)
    return names


def _ThemeKeywordHits(board_name: str, keywords: list[str]) -> int:
    """统计板块名称与题材关键词的命中数。"""
    text = board_name.lower()
    hits = 0
    for kw in keywords:
        token = kw.strip().lower()
        if len(token) >= 2 and token in text:
            hits += 1
    return hits


def _ExtractMainNetInflow(data: dict[str, Any] | None) -> float:
    """提取最近一日主力净流入。"""
    fund_flow = None
    if data:
        fund_flow = data.get("fund_flow")
    if fund_flow is None:
        return 0.0
    try:
        if getattr(fund_flow, "empty", True):
            return 0.0
        col = next(
            (c for c in fund_flow.columns if "主力净流入" in str(c) and "净额" in str(c)),
            None,
        )
        if col is None:
            return 0.0
        return SafeFloat(fund_flow[col].iloc[-1])
    except Exception:
        return 0.0


def ComputeSectorThemeOverlap(data: dict[str, Any] | None) -> dict[str, Any]:
    """计算今日概念领涨与配置题材的匹配度。"""
    themes = [t.strip() for t in config.STOCK_THEMES if t.strip()]
    top_names = _ExtractConceptTopNames(data)
    if not top_names:
        return {
            "overlap_level": "未知",
            "overlap_hits": 0,
            "top_concepts": [],
            "matched_concepts": [],
        }

    matched: list[str] = []
    total_hits = 0
    for name in top_names:
        hits = _ThemeKeywordHits(name, themes) if themes else 0
        if hits > 0:
            matched.append(name)
            total_hits += hits

    if not themes:
        level = "未知"
    elif total_hits >= _OVERLAP_HIGH_THRESHOLD:
        level = "高"
    elif total_hits >= _OVERLAP_MED_THRESHOLD:
        level = "中"
    else:
        level = "低"

    return {
        "overlap_level": level,
        "overlap_hits": total_hits,
        "top_concepts": top_names,
        "matched_concepts": matched,
    }


def ComputeStockVsSectorDivergence(data: dict[str, Any] | None) -> float:
    """个股涨跌幅减概念板块 TOP5 均值（正=强于板块）。"""
    if not data:
        return 0.0
    rt = data.get("realtime") or {}
    stock_pct = SafeFloat(rt.get("change_pct"))
    concept = data.get("concept")
    if concept is None or getattr(concept, "empty", True):
        return stock_pct
    pcts: list[float] = []
    for i in range(min(5, len(concept))):
        pcts.append(SafeFloat(concept.iloc[i].get("涨跌幅")))
    if not pcts:
        return stock_pct
    sector_avg = sum(pcts) / len(pcts)
    return round(stock_pct - sector_avg, 2)


def BuildCompanyNewsContext(
    data: dict[str, Any] | None,
    news_bundle: dict[str, Any] | None = None,
) -> list[str]:
    """构建资讯评分用的公司/盘面上下文。"""
    lines: list[str] = [
        f"目标股票：{config.STOCK_NAME}({config.STOCK_CODE})",
    ]
    if config.STOCK_BUSINESS:
        lines.append(f"公司主业/业绩：{config.STOCK_BUSINESS}")
    if config.STOCK_THEMES:
        lines.append(f"配置题材：{', '.join(config.STOCK_THEMES)}")

    overlap = ComputeSectorThemeOverlap(data)
    lines.append(
        f"概念领涨与题材重叠度：{overlap.get('overlap_level', '未知')}"
        f"（命中 {overlap.get('overlap_hits', 0)}）"
    )
    if overlap.get("top_concepts"):
        lines.append(f"今日概念TOP：{', '.join(overlap['top_concepts'][:5])}")
    if overlap.get("matched_concepts"):
        lines.append(f"与题材匹配板块：{', '.join(overlap['matched_concepts'])}")

    rt = data.get("realtime") or {} if data else {}
    if rt:
        divergence = ComputeStockVsSectorDivergence(data)
        main_net = _ExtractMainNetInflow(data)
        lines.extend([
            f"个股涨跌幅：{SafeFloat(rt.get('change_pct')):+.2f}%",
            f"相对概念均值偏离：{divergence:+.2f}%（负=弱于板块）",
            f"主力净流入：{main_net:.0f}元",
            f"市盈率：{SafeFloat(rt.get('pe')):.1f}",
        ])

    bundle = news_bundle or {}
    snapshot = bundle.get("sector_snapshot") or []
    if snapshot:
        lines.append("")
        lines.extend(snapshot)

    return lines
