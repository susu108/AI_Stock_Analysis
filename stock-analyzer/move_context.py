"""涨跌归因上下文 — 聚合行情/技术/资金/资讯结构化事实。"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

import config
from news_analyzer import GetBackgroundNewsItems, GetImpactfulItems
from news_context import ComputeSectorThemeOverlap, ComputeStockVsSectorDivergence
from price_context import CalcPriceAnchors, ExtractRecentPriceLevels
from utils import FormatAmount, SafeFloat, SetupLogger

logger = SetupLogger(config.LOG_LEVEL)


def ClassifyMarketCapTier(circ_mv: float) -> str:
    """根据流通市值划分盘子大小。"""
    if circ_mv <= 0:
        return "未知"
    if circ_mv < 30e8:
        return "小盘（<30亿）"
    if circ_mv < 100e8:
        return "中盘（30~100亿）"
    return "大盘（>100亿）"


def CalcPrevDayChangePct(kline: pd.DataFrame | None) -> float:
    """计算昨日涨跌幅。"""
    if kline is None or kline.empty or len(kline) < 2:
        return 0.0
    pct_col = "涨跌幅" if "涨跌幅" in kline.columns else None
    if pct_col:
        return SafeFloat(kline[pct_col].iloc[-2])
    close_col = "收盘" if "收盘" in kline.columns else "close"
    closes = kline[close_col].astype(float)
    prev = closes.iloc[-2]
    if prev <= 0:
        return 0.0
    return round((closes.iloc[-1] - prev) / prev * 100, 2)


def BuildBoxBreakNarrative(
    price: float,
    levels: dict[str, float],
    anchors: dict[str, float],
) -> str:
    """生成箱体破位/压力转换叙事。"""
    box_low = anchors.get("box_low", 0.0)
    box_high = anchors.get("box_high", 0.0)
    prev_low = levels.get("prev_day_low", 0.0)
    today_open = levels.get("today_open", price)
    parts: list[str] = []

    if box_low > 0 and box_high > box_low:
        parts.append(f"近10日箱体 {box_low:.2f}~{box_high:.2f} 元")
        if prev_low > 0 and prev_low < box_low:
            parts.append(f"前一日最低 {prev_low:.2f} 元跌破箱体下沿，短线抛压释放")
        if price > box_high:
            parts.append(f"现价 {price:.2f} 元放量突破箱体上沿，原压力或转支撑")
        elif price < box_low:
            parts.append(f"现价位于箱体下沿下方，属超跌区域")

    if today_open > 0 and price > today_open * 1.01:
        parts.append(f"开盘 {today_open:.2f} 元后上攻，日内强势")
    elif today_open > 0 and price < today_open * 0.99:
        parts.append(f"低开 {today_open:.2f} 元后走弱")

    return "；".join(parts) if parts else "暂无显著箱体破位信号"


def ExtractSectorLeaders(data: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    """提取概念/行业板块领涨领跌。"""
    leaders: list[dict[str, Any]] = []
    for board_key, label in (("concept", "概念"), ("industry", "行业")):
        board = data.get(board_key)
        if board is None or getattr(board, "empty", True):
            continue
        sorted_board = board.sort_values("涨跌幅", ascending=False)
        for i in range(min(limit, len(sorted_board))):
            row = sorted_board.iloc[i]
            leaders.append({
                "type": label,
                "name": str(row.get("板块名称", "")),
                "change_pct": SafeFloat(row.get("涨跌幅")),
            })
    return leaders


def ExtractMainNetInflow(fund_flow: pd.DataFrame | None) -> float:
    """提取最近一日主力净流入。"""
    if fund_flow is None or fund_flow.empty:
        return 0.0
    col = next(
        (c for c in fund_flow.columns if "主力净流入" in str(c) and "净额" in str(c)),
        None,
    )
    if col is None:
        return 0.0
    return SafeFloat(fund_flow[col].iloc[-1])


def FormatNewsForContext(items: list[dict[str, Any]], limit: int = 8) -> list[dict[str, str]]:
    """格式化资讯供 LLM 引用。"""
    formatted: list[dict[str, str]] = []
    for item in items[:limit]:
        formatted.append({
            "category": str(item.get("category", "")),
            "title": str(item.get("title", "")),
            "content": str(item.get("content", ""))[:300],
            "impact": str(item.get("impact", "")),
            "directness": str(item.get("directness", "")),
            "impact_reason": str(item.get("impact_reason", "")),
            "source": str(item.get("source", "")),
            "url": str(item.get("url", "")),
        })
    return formatted


def BuildMoveContext(
    data: dict[str, Any],
    analysis: dict[str, Any],
    news_bundle: dict[str, Any] | None,
) -> dict[str, Any]:
    """聚合涨跌归因所需的结构化上下文。"""
    rt = data.get("realtime") or {}
    kline = data.get("kline")
    indicators = analysis.get("indicators", {})
    bundle = news_bundle or {}

    price = SafeFloat(rt.get("price") or indicators.get("price"))
    change_pct = SafeFloat(rt.get("change_pct"))
    change_amt = SafeFloat(rt.get("change_amt"))
    prev_close = SafeFloat(rt.get("prev_close"))
    amount = SafeFloat(rt.get("amount"))
    volume_ratio = SafeFloat(rt.get("volume_ratio", 1.0))
    circ_mv = SafeFloat(rt.get("circ_mv"))
    total_mv = SafeFloat(rt.get("total_mv"))

    levels = ExtractRecentPriceLevels(kline, rt)
    anchors = CalcPriceAnchors(kline, rt, indicators, price)
    prev_day_change = CalcPrevDayChangePct(kline)

    margin = data.get("margin") or {}
    margin_change = SafeFloat(margin.get("margin_balance_change"))
    margin_balance = SafeFloat(margin.get("margin_balance"))

    relevant = list(bundle.get("relevant_items") or [])
    web_hits = list(bundle.get("web_search") or [])
    impactful = GetImpactfulItems(relevant)
    background = GetBackgroundNewsItems(relevant)

    overlap = ComputeSectorThemeOverlap(data)
    divergence = ComputeStockVsSectorDivergence(data)

    move_direction = "平"
    if change_pct > 0.3:
        move_direction = "涨"
    elif change_pct < -0.3:
        move_direction = "跌"

    context: dict[str, Any] = {
        "stock_name": config.STOCK_NAME,
        "stock_code": config.STOCK_CODE,
        "stock_themes": list(config.STOCK_THEMES),
        "stock_business": config.STOCK_BUSINESS,
        "sector_theme_overlap": overlap,
        "stock_vs_sector_divergence": divergence,
        "move_date": date.today().isoformat(),
        "move_direction": move_direction,
        "change_pct": change_pct,
        "change_amt": change_amt,
        "price": price,
        "prev_close": prev_close,
        "amount": amount,
        "amount_text": FormatAmount(amount),
        "volume_ratio": volume_ratio,
        "prev_day_change_pct": prev_day_change,
        "prev_day_low": levels.get("prev_day_low", 0.0),
        "today_low": levels.get("today_low", 0.0),
        "today_high": levels.get("today_high", 0.0),
        "today_open": levels.get("today_open", 0.0),
        "circ_mv": circ_mv,
        "circ_mv_yi": round(circ_mv / 1e8, 2) if circ_mv > 0 else 0.0,
        "total_mv": total_mv,
        "mv_tier": ClassifyMarketCapTier(circ_mv),
        "box_low": anchors.get("box_low", 0.0),
        "box_high": anchors.get("box_high", 0.0),
        "box_mid": anchors.get("box_mid", 0.0),
        "box_break_narrative": BuildBoxBreakNarrative(price, levels, anchors),
        "sector_leaders": ExtractSectorLeaders(data),
        "sector_snapshot": list(bundle.get("sector_snapshot") or []),
        "main_net_inflow": ExtractMainNetInflow(data.get("fund_flow")),
        "margin_balance": margin_balance,
        "margin_balance_change": margin_change,
        "margin_buy": SafeFloat(margin.get("margin_buy")),
        "tech_signals": list(analysis.get("tech_signals") or []),
        "fund_signals": list(analysis.get("fund_signals") or []),
        "sector_signals": list(analysis.get("sector_signals") or []),
        "news_signals": list(analysis.get("news_signals") or []),
        "relevant_news": FormatNewsForContext(relevant),
        "impactful_news": FormatNewsForContext(impactful, limit=6),
        "background_news": FormatNewsForContext(background, limit=6),
        "web_search_hits": FormatNewsForContext(web_hits, limit=6),
        "news_score": SafeFloat(analysis.get("news_score")),
        "weighted_score": SafeFloat(analysis.get("weighted_score")),
    }

    logger.debug(
        "涨跌上下文 — %s %+.2f%% 量比%.2f 流通市值%.1f亿 融资变动%.0f",
        move_direction,
        change_pct,
        volume_ratio,
        context["circ_mv_yi"],
        margin_change,
    )
    return context
