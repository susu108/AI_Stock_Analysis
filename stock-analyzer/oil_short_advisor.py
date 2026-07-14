"""石油短线实操 — 规则兜底与 news_alert 聚合。"""

from __future__ import annotations

from typing import Any

import config
from oil_short_playbook import (
    MaxPositionPctForStock,
    OIL_SHORT_RISK_HEADER,
)
from utils import SafeFloat


def AggregateOilNewsAlert(
    news_bundle: dict[str, Any] | None,
    news_score: float = 0.0,
) -> dict[str, Any]:
    """从资讯包聚合最强利好/利空警报。"""
    bundle = news_bundle or {}
    items = list(bundle.get("relevant_items") or [])
    best: dict[str, Any] | None = None
    best_score = 0

    for item in items:
        impact = str(item.get("impact", ""))
        if impact not in ("涨", "跌"):
            continue
        strength = int(SafeFloat(item.get("strength", 0)) or 0)
        sign = 1 if impact == "涨" else -1
        score = sign * strength * 20
        if best is None or abs(score) > abs(best_score):
            best = item
            best_score = score

    if best is None:
        if abs(news_score) >= 50:
            level = "strong_bull" if news_score > 0 else "strong_bear"
            return {
                "level": level,
                "score": int(round(news_score)),
                "headline": "综合资讯面偏强" if news_score > 0 else "综合资讯面偏弱",
                "impact_on_advice": (
                    "资讯偏多但仍须遵守不追高与仓位红线"
                    if news_score > 0
                    else "资讯偏空，优先观望/止损，降低仓位"
                ),
            }
        return {
            "level": "none",
            "score": 0,
            "headline": "",
            "impact_on_advice": "",
        }

    strength = int(SafeFloat(best.get("strength", 0)) or 0)
    impact = str(best.get("impact", ""))
    title = str(best.get("title", "")).strip()
    if strength >= 4 or abs(best_score) >= 80:
        level = "strong_bull" if impact == "涨" else "strong_bear"
    elif strength >= 2:
        level = "mild_bull" if impact == "涨" else "mild_bear"
    else:
        level = "none"

    if level == "strong_bear":
        impact_text = "强烈利空，建议观望或止损离场，仓位趋近0"
    elif level == "strong_bull":
        impact_text = "强烈利好但若当日涨幅已大优先反T减仓，禁追高开仓"
    elif level == "mild_bear":
        impact_text = "偏空，控制仓位、严守止损"
    elif level == "mild_bull":
        impact_text = "偏多，仅回调支撑区间分批低吸"
    else:
        impact_text = ""

    return {
        "level": level,
        "score": int(best_score),
        "headline": title[:80],
        "impact_on_advice": impact_text,
        "source_title": title,
        "strength": strength,
    }


def BuildRuleShortPlayFallback(
    advice: dict[str, Any],
    analysis: dict[str, Any],
    data: dict[str, Any],
    news_bundle: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """LLM 不可用时生成 short_play + news_alert。"""
    rt = data.get("realtime") or {}
    price = SafeFloat(rt.get("price") or analysis.get("indicators", {}).get("price"))
    change_pct = SafeFloat(rt.get("change_pct"))
    zones = advice.get("trade_zones") or {}
    at1 = zones.get("add_tier1") or {}
    at2 = zones.get("add_tier2") or {}
    st1 = zones.get("sell_tier1") or {}
    st2 = zones.get("sell_tier2") or {}

    entry_low = SafeFloat(at1.get("low") or advice.get("buy_low"))
    entry_high = SafeFloat(at2.get("high") or advice.get("buy_high") or at1.get("high"))
    if entry_low <= 0 and price > 0:
        entry_low = round(price * 0.97, 2)
    if entry_high <= 0 and price > 0:
        entry_high = round(price * 0.99, 2)

    stop = SafeFloat(at1.get("stop_loss") or advice.get("stop_loss"))
    if stop <= 0 and entry_low > 0:
        stop = round(entry_low - 0.5, 2)
    if stop <= 0 and price > 0:
        stop = round(price * 0.95, 2)

    mid_entry = (entry_low + entry_high) / 2 if entry_low and entry_high else price
    tp1 = SafeFloat(st1.get("best") or st1.get("low"))
    if tp1 <= 0 and mid_entry > 0:
        tp1 = round(mid_entry * 1.04, 2)
    tp2 = SafeFloat(st2.get("best") or st2.get("high") or st1.get("high"))
    if tp2 <= 0 and mid_entry > 0:
        tp2 = round(mid_entry * 1.08, 2)

    news_score = SafeFloat(analysis.get("news_score"))
    alert = AggregateOilNewsAlert(news_bundle, news_score)
    max_pct = MaxPositionPctForStock(config.STOCK_CODE)

    mode = "波段1-3天"
    tactic = "波段低吸"
    hold_days = "1-3天"
    buy_action = "轻仓买入"
    position_pct = min(max_pct * 0.5, max_pct)

    if alert["level"] == "strong_bear":
        mode, tactic, hold_days = "空仓", "空仓观望", "不建议持股"
        buy_action = "观望"
        position_pct = 0.0
    elif change_pct > 4:
        mode, tactic, hold_days = "日内T", "反T", "当日"
        buy_action = "观望"
        position_pct = 0.0
    elif change_pct < -2:
        mode, tactic = "日内T", "正T"
        hold_days = "当日"
        buy_action = "轻仓买入"
        position_pct = min(max_pct * 0.5, max_pct)
    elif alert["level"] == "strong_bull" and change_pct > 2:
        mode, tactic, hold_days = "日内T", "反T", "当日"
        buy_action = "观望"
        position_pct = 0.0

    short_play = {
        "mode": mode,
        "position_pct": round(position_pct, 1),
        "entry_low": round(entry_low, 2),
        "entry_high": round(entry_high, 2),
        "stop_loss": round(stop, 2),
        "take_profit_1": round(tp1, 2),
        "take_profit_2": round(tp2, 2),
        "hold_days": hold_days,
        "tactic": tactic,
        "reasons": [
            f"规则兜底：现价{price:.2f}，涨跌{change_pct:+.2f}%",
            alert.get("impact_on_advice") or "结合支撑压力区间短线操作",
            OIL_SHORT_RISK_HEADER,
        ],
        "discipline": [
            "油气赛道合计仓位≤20%，遵守单票仓位红线",
            "单日涨幅>4%禁止新开仓，只做反T",
            "日内T当日清仓；波段硬止损5%",
            "地缘缓和/原油大跌优先减仓离场",
        ],
    }
    advice_overrides = {
        "buy_action": buy_action,
        "sell_action": "减仓" if alert["level"] == "strong_bear" or change_pct > 4 else "持有",
    }
    return short_play, {**alert, **{"_advice_overrides": advice_overrides}}


def ParseShortPlayFromLlm(
    llm_result: dict[str, Any],
    fallback: dict[str, Any],
) -> dict[str, Any]:
    """从 LLM 结果解析 short_play，缺失字段用兜底填充。"""
    raw = llm_result.get("short_play")
    if not isinstance(raw, dict):
        return dict(fallback)
    result = dict(fallback)
    for key in (
        "mode", "tactic", "hold_days",
    ):
        val = str(raw.get(key, "")).strip()
        if val:
            result[key] = val
    for key in (
        "position_pct", "entry_low", "entry_high",
        "stop_loss", "take_profit_1", "take_profit_2",
    ):
        num = SafeFloat(raw.get(key))
        if num != 0 or key == "position_pct":
            if key == "position_pct" or num > 0:
                result[key] = round(num, 2 if key != "position_pct" else 1)

    reasons = raw.get("reasons")
    if isinstance(reasons, list) and reasons:
        result["reasons"] = [str(r).strip() for r in reasons if str(r).strip()][:4]
    discipline = raw.get("discipline")
    if isinstance(discipline, list) and discipline:
        result["discipline"] = [str(d).strip() for d in discipline if str(d).strip()][:5]

    max_pct = MaxPositionPctForStock()
    if SafeFloat(result.get("position_pct")) > max_pct:
        result["position_pct"] = max_pct
    return result


def ParseNewsAlertFromLlm(
    llm_result: dict[str, Any],
    fallback: dict[str, Any],
) -> dict[str, Any]:
    """从 LLM 解析 news_alert。"""
    raw = llm_result.get("news_alert")
    if not isinstance(raw, dict):
        return dict(fallback)
    level = str(raw.get("level", fallback.get("level", "none"))).strip()
    valid = {
        "strong_bull", "strong_bear", "mild_bull", "mild_bear", "none",
    }
    if level not in valid:
        level = str(fallback.get("level", "none"))
    return {
        "level": level,
        "score": int(SafeFloat(raw.get("score", fallback.get("score", 0)))),
        "headline": str(raw.get("headline", "")).strip()
        or str(fallback.get("headline", "")),
        "impact_on_advice": str(raw.get("impact_on_advice", "")).strip()
        or str(fallback.get("impact_on_advice", "")),
    }


def ApplyOilNewsConstraints(
    short_play: dict[str, Any],
    news_alert: dict[str, Any],
    change_pct: float,
) -> tuple[dict[str, Any], str, str]:
    """按强讯与涨幅约束仓位与买卖动作。"""
    play = dict(short_play)
    buy_action = "轻仓买入"
    sell_action = "持有"
    level = str(news_alert.get("level", "none"))

    if level == "strong_bear":
        play["mode"] = "空仓"
        play["tactic"] = "空仓观望"
        play["hold_days"] = "不建议持股"
        play["position_pct"] = 0.0
        buy_action, sell_action = "观望", "减仓"
    elif change_pct > 4:
        play["mode"] = "日内T"
        play["tactic"] = "反T"
        play["hold_days"] = "当日"
        play["position_pct"] = 0.0
        buy_action, sell_action = "观望", "减仓"
    elif level == "strong_bull" and change_pct > 2:
        play["tactic"] = "反T"
        play["position_pct"] = 0.0
        buy_action = "观望"
        sell_action = "减仓"

    max_pct = MaxPositionPctForStock()
    if SafeFloat(play.get("position_pct")) > max_pct:
        play["position_pct"] = max_pct
    return play, buy_action, sell_action
