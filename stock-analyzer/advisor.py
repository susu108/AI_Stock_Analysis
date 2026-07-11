"""买卖建议模块 — 支撑位/压力位/止损/策略/推荐理由。"""

from __future__ import annotations

import math
from typing import Any

from llm_advisor import EnhanceAdviceWithLlm
from market_context import ResolvePredictionHorizon
from price_context import BuildPriceContext, ExtractRecentPriceLevels
from trade_zones import BuildRuleTradePlan, EnvelopeFromTradeZones
from utils import SetupLogger
import config

logger = SetupLogger(config.LOG_LEVEL)

BUY_KEYWORDS = ("金叉", "多头排列", "超卖", "支撑", "放量上涨", "站上MA20", "持续流入", "净流入")
SELL_KEYWORDS = ("死叉", "空头排列", "超买", "压力", "放量下跌", "绿柱", "持续流出", "净流出", "净卖出", "跌破", "仅")


def _PickDeepSupport(
    price: float,
    recent: dict[str, float],
    low_20: float,
    boll_lower: float,
) -> float:
    """第二支撑位 S2：昨日/近3~5日真实低点为主，布林/20日低为辅。"""
    recent_lows = [
        recent.get("prev_day_low", 0.0),
        recent.get("low_3", 0.0),
        recent.get("low_5", 0.0),
    ]
    recent_below = sorted({round(v, 2) for v in recent_lows if 0 < v < price})
    if recent_below:
        return recent_below[0]

    fallback = sorted(
        {round(v, 2) for v in [low_20, boll_lower] if 0 < v < price},
    )
    if fallback:
        return fallback[0]
    return round(price * 0.95, 2)


def _PickNearSupport(
    price: float,
    s2: float,
    recent: dict[str, float],
    ma5: float,
    ma10: float,
) -> float:
    """第一支撑位 S1：优先今日低/今开，其次均线。"""
    action_levels = [
        recent.get("today_low", 0.0),
        recent.get("today_open", 0.0),
    ]
    action_between = sorted(
        {round(v, 2) for v in action_levels if s2 < v < price},
        reverse=True,
    )
    if action_between:
        return action_between[0]

    fallback_candidates = [
        recent.get("prev_day_low", 0.0),
        ma5,
        ma10,
    ]
    between = sorted(
        {round(v, 2) for v in fallback_candidates if s2 < v < price},
        reverse=True,
    )
    if between:
        return between[0]
    if s2 < price:
        return round(min(price - 0.01, s2 * 1.03), 2)
    return round(price * 0.98, 2)


def _PickResistances(
    price: float,
    recent: dict[str, float],
    ma20: float,
    boll_mid: float,
    boll_upper: float,
    high_20: float,
) -> tuple[float, float]:
    """压力位：优先今日高/近端高点，远端布林/20日高作 R2。"""
    near_resist = [
        recent.get("today_high", 0.0),
        recent.get("high_3", 0.0),
        recent.get("high_5", 0.0),
    ]
    far_resist = [ma20, boll_mid, boll_upper, high_20]

    near_above = sorted({round(v, 2) for v in near_resist if v > price})
    far_above = sorted({round(v, 2) for v in far_resist if v > price})

    if near_above:
        r1 = near_above[0]
    elif far_above:
        r1 = far_above[0]
    else:
        r1 = round(price * 1.02, 2)

    all_above = sorted(set(near_above + far_above))
    higher = [v for v in all_above if v > r1]
    if higher:
        r2 = higher[0]
    elif far_above and far_above[0] > r1:
        r2 = far_above[0]
    else:
        r2 = round(max(r1 * 1.03, float(math.ceil(r1))), 2)

    if r2 < r1:
        r2 = r1
    return r1, r2


def CalcSupportResistance(
    indicators: dict[str, Any],
    realtime: dict[str, Any] | None = None,
) -> dict[str, float]:
    """计算支撑位和压力位（结合近期 K 线真实高低点）。"""
    if not indicators:
        price = 0.0
        return {
            "s1": price,
            "s2": price,
            "r1": price,
            "r2": price,
            "stop_loss": price,
        }

    ma5 = indicators.get("ma5", 0.0)
    ma10 = indicators.get("ma10", 0.0)
    ma20 = indicators.get("ma20", 0.0)
    boll_mid = indicators.get("boll_mid", 0.0)
    boll_upper = indicators.get("boll_upper", 0.0)
    boll_lower = indicators.get("boll_lower", 0.0)
    low_20 = indicators.get("low_20", 0.0)
    high_20 = indicators.get("high_20", 0.0)
    price = indicators.get("price", 0.0)
    kline = indicators.get("kline")

    if price <= 0:
        return {
            "s1": 0.0,
            "s2": 0.0,
            "r1": 0.0,
            "r2": 0.0,
            "stop_loss": 0.0,
            "price": 0.0,
        }

    recent = ExtractRecentPriceLevels(kline, realtime)
    s2 = _PickDeepSupport(price, recent, low_20, boll_lower)
    s1 = _PickNearSupport(price, s2, recent, ma5, ma10)
    if s2 >= s1:
        s1, s2 = s2, round(s1 * 0.97, 2)

    r1, r2 = _PickResistances(price, recent, ma20, boll_mid, boll_upper, high_20)

    if s1 >= r1:
        gap = price * 0.02
        s1 = round(price - gap, 2)
        s2 = round(price - gap * 2, 2)
        r1 = round(price + gap, 2)
        r2 = round(max(price + gap * 2, float(math.ceil(price + gap * 2))), 2)

    stop_loss = round(s2 * 0.98, 2)

    return {
        "s1": round(s1, 2),
        "s2": round(s2, 2),
        "r1": round(r1, 2),
        "r2": round(r2, 2),
        "stop_loss": stop_loss,
        "price": price,
    }


def GenerateStrategy(weighted_score: float) -> str:
    """根据加权分数生成操作策略。"""
    if weighted_score > 30:
        return "整体偏强，可逢回调至支撑位分批低吸，设好止损"
    if weighted_score > 10:
        return "温和偏多，轻仓试探性买入，关注量能配合"
    if weighted_score > -10:
        return "方向不明，建议观望为主，等待趋势明确"
    if weighted_score > -30:
        return "温和偏空，持仓者可逢反弹至压力位减仓"
    return "整体偏弱，建议规避或轻仓，不宜追涨"


def ExtractReasons(
    tech_signals: list[str],
    fund_signals: list[str],
    keywords: tuple[str, ...],
    max_count: int = 3,
) -> list[str]:
    """从信号中提取包含关键词的推荐理由。"""
    all_signals = [f"技术面：{s}" for s in tech_signals] + [
        f"资金面：{s}" for s in fund_signals
    ]
    matched: list[str] = []
    for signal in all_signals:
        if any(kw in signal for kw in keywords):
            if "仅" in signal and "净买入" in signal:
                continue
            matched.append(signal)
            if len(matched) >= max_count:
                break
    return matched


def GenerateAdvice(
    analysis: dict[str, Any],
    data: dict[str, Any],
    news_items: list[dict[str, str]] | None = None,
    news_bundle: dict[str, Any] | None = None,
    session_label: str = "盘中",
    mode: str = "daily",
    prediction_horizon: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """生成完整买卖建议。"""
    indicators = analysis.get("indicators", {})
    rt = data.get("realtime") or {}
    current_price = float(rt.get("price") or indicators.get("price", 0))
    if current_price > 0:
        indicators = {**indicators, "price": current_price}

    sr = CalcSupportResistance(indicators, realtime=rt)

    tech_score = analysis.get("tech_score", 0.0)
    fund_score = analysis.get("fund_score", 0.0)
    weighted = analysis.get("weighted_score", 0.0)

    buy_ok = tech_score + fund_score > 0
    sell_ok = tech_score + fund_score < 0

    buy_reasons = ExtractReasons(
        analysis.get("tech_signals", []),
        analysis.get("fund_signals", []),
        BUY_KEYWORDS,
    )
    sell_reasons = ExtractReasons(
        analysis.get("tech_signals", []),
        analysis.get("fund_signals", []),
        SELL_KEYWORDS,
    )

    if not buy_reasons:
        buy_reasons = analysis.get("tech_signals", [])[:2]
    if not sell_reasons:
        sell_reasons = [
            s for s in analysis.get("tech_signals", []) if "超买" in s or "压力" in s
        ][:2]

    strategy = GenerateStrategy(weighted)
    price_ctx = BuildPriceContext(
        data.get("kline"), indicators, current_price, realtime=rt,
    )

    rule_plan = BuildRuleTradePlan(price_ctx, sr, current_price, analysis)
    rule_zones = {
        "add_tier1": rule_plan["add_tier1"],
        "add_tier2": rule_plan["add_tier2"],
        "sell_tier1": rule_plan["sell_tier1"],
        "sell_tier2": rule_plan["sell_tier2"],
    }
    envelope = EnvelopeFromTradeZones(rule_zones)
    tier1_stop = float(rule_plan["add_tier1"].get("stop_loss", 0) or sr["stop_loss"])

    logger.info(
        "买卖建议 — 加一:%.2f~%.2f 加二:%.2f~%.2f 卖一:%.2f~%.2f 卖二:%.2f~%.2f 止损:%.2f"
        "（箱体:%.2f VWAP:%.2f 禁加:%.2f）",
        rule_zones["add_tier1"]["low"],
        rule_zones["add_tier1"]["high"],
        rule_zones["add_tier2"]["low"],
        rule_zones["add_tier2"]["high"],
        rule_zones["sell_tier1"]["low"],
        rule_zones["sell_tier1"]["high"],
        rule_zones["sell_tier2"]["low"],
        rule_zones["sell_tier2"]["high"],
        tier1_stop,
        price_ctx.get("box_mid", 0),
        price_ctx.get("prev_day_vwap", 0),
        price_ctx.get("forbidden_add_above", 0),
    )

    base_advice = {
        "s1": sr["s1"],
        "s2": sr["s2"],
        "r1": sr["r1"],
        "r2": sr["r2"],
        "stop_loss": tier1_stop,
        "buy_low": envelope["buy_low"],
        "buy_high": envelope["buy_high"],
        "sell_low": envelope["sell_low"],
        "sell_high": envelope["sell_high"],
        "trade_plan": rule_plan,
        "trade_zones": rule_zones,
        "buy_ok": buy_ok,
        "sell_ok": sell_ok,
        "strategy": strategy,
        "buy_reasons": buy_reasons,
        "sell_reasons": sell_reasons,
        "rule_s1": sr["s1"],
        "rule_s2": sr["s2"],
        "rule_r1": sr["r1"],
        "rule_r2": sr["r2"],
        "prediction_horizon": prediction_horizon or ResolvePredictionHorizon(session_label),
    }

    return EnhanceAdviceWithLlm(
        analysis,
        data,
        base_advice,
        price_ctx,
        news_items or [],
        session_label,
        news_bundle=news_bundle,
        mode=mode,
    )
