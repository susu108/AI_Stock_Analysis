"""分档加仓/卖出计划 — 规则构建与 LLM 解析。"""

from __future__ import annotations

from typing import Any, Literal

Side = Literal["buy", "sell"]

_DEFAULT_RISK = (
    "以下价格仅为技术区间参考，不构成投资建议；"
    "请结合个股题材属性与基本面风险，控制仓位，禁止重仓长期加仓。"
)


def _SafeFloat(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _RoundPrice(value: float) -> float:
    return round(value, 2)


def _NarrowBand(center: float, pct: float = 0.004, min_half: float = 0.15) -> tuple[float, float]:
    """以中心价生成窄区间。"""
    if center <= 0:
        return 0.0, 0.0
    half = max(center * pct, min_half)
    return _RoundPrice(center - half), _RoundPrice(center + half)


def _IsBuyZoneValid(low: float, high: float, current_price: float) -> bool:
    return low > 0 and high > 0 and low <= high < current_price


def _IsSellZoneValid(low: float, high: float, current_price: float) -> bool:
    return low > 0 and high > 0 and current_price < low <= high


def _BuildTier(
    low: float,
    high: float,
    label: str = "",
    logic: str = "",
    rules: list[str] | None = None,
    stop_loss: float = 0.0,
    note: str = "",
    anchor: float = 0.0,
    recommended: float = 0.0,
    recommended_source: str = "",
) -> dict[str, Any]:
    if low > high and high > 0:
        low, high = high, low
    lo = _RoundPrice(low)
    hi = _RoundPrice(high)
    anc = _RoundPrice(anchor) if anchor > 0 else 0.0
    rec = _RoundPrice(recommended) if recommended > 0 else 0.0
    if rec > 0 and lo > 0 and hi > 0:
        rec = max(lo, min(hi, rec))
    elif anc > 0 and lo > 0 and hi > 0:
        rec = max(lo, min(hi, anc))
    return {
        "low": lo,
        "high": hi,
        "label": label.strip()[:40],
        "logic": logic.strip()[:120],
        "rules": [str(r).strip()[:60] for r in (rules or []) if str(r).strip()][:4],
        "stop_loss": _RoundPrice(stop_loss) if stop_loss > 0 else 0.0,
        "note": note.strip()[:80] or logic.strip()[:80],
        "anchor": anc,
        "recommended": rec,
        "recommended_source": recommended_source.strip()[:16],
    }


def _EmptyTier() -> dict[str, Any]:
    return _BuildTier(0.0, 0.0)


def ResolveTierRecommended(tier: dict[str, Any], side: Side) -> float:
    """解析档位最推荐价位（AI/锚点优先，宽区间时偏深支撑或近端压力）。"""
    low = _SafeFloat(tier.get("low"))
    high = _SafeFloat(tier.get("high"))
    if low <= 0 or high <= 0:
        return 0.0
    if low > high:
        low, high = high, low
    rec = _SafeFloat(tier.get("recommended"))
    if rec > 0 and low <= rec <= high:
        return rec
    anchor = _SafeFloat(tier.get("anchor"))
    if anchor > 0 and low <= anchor <= high:
        return anchor
    if side == "buy":
        return _RoundPrice(low + (high - low) * 0.35)
    return _RoundPrice(low + (high - low) * 0.45)


def BuildRuleTradePlan(
    price_ctx: dict[str, Any],
    sr: dict[str, float],
    current_price: float,
    analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """基于技术锚点构建完整分档交易计划（豆包维度）。"""
    if current_price <= 0:
        empty = _EmptyTier()
        return {
            "risk_warning": _DEFAULT_RISK,
            "add_tier1": empty,
            "add_tier2": empty,
            "sell_tier1": empty,
            "sell_tier2": empty,
            "no_add_zones": [],
            "discipline": [],
            "breakdown_plan": {"level": 0.0, "next_support": 0.0, "action": ""},
        }

    box_mid = _SafeFloat(price_ctx.get("box_mid"))
    box_low = _SafeFloat(price_ctx.get("box_low"))
    prev_vwap = _SafeFloat(price_ctx.get("prev_day_vwap"))
    ma5 = _SafeFloat(price_ctx.get("ma5"))
    key_support = _SafeFloat(price_ctx.get("key_support"))
    next_support = _SafeFloat(price_ctx.get("next_support_if_break"))
    forbidden = _SafeFloat(price_ctx.get("forbidden_add_above"))
    pressure_t1 = _SafeFloat(price_ctx.get("pressure_t1"))
    pressure_t2 = _SafeFloat(price_ctx.get("pressure_t2"))
    rally_low = _SafeFloat(price_ctx.get("rally_start_low"))

    # 第一加仓（深支撑/箱体中枢，优先等待）
    t1_center = box_mid if box_mid > 0 else (box_low if box_low > 0 else key_support)
    if rally_low > 0 and rally_low < t1_center:
        t1_center = round((t1_center + rally_low) / 2, 2)
    t1_lo, t1_hi = _NarrowBand(t1_center, pct=0.005, min_half=0.25)
    if key_support > 0 and t1_lo > key_support:
        t1_lo = _RoundPrice(key_support)
    t1_hi = min(t1_hi, current_price - 0.01)
    if t1_hi <= t1_lo:
        t1_lo, t1_hi = _NarrowBand(t1_center, pct=0.008, min_half=0.3)
        t1_hi = min(t1_hi, current_price - 0.01)

    add_tier1 = _BuildTier(
        t1_lo,
        t1_hi,
        label="轻仓试错第一加仓（优先等待）",
        logic="近端箱体中枢/拉升启动区，短线筹码成本集中，回踩承接较强",
        rules=[
            "跌到区间内、缩量止跌（小阴/十字星）再加",
            "单次小仓位试探，纯做波段降本",
        ],
        stop_loss=key_support if key_support > 0 else _RoundPrice(t1_lo * 0.98),
        anchor=t1_center,
        recommended=t1_center,
        recommended_source="rule",
    )

    # 第二加仓（浅回调/VWAP，次选）
    t2_center = prev_vwap if prev_vwap > 0 else ma5
    if t2_center <= 0:
        t2_center = _RoundPrice((add_tier1["high"] + current_price) / 2)
    t2_lo, t2_hi = _NarrowBand(t2_center, pct=0.003, min_half=0.15)
    if t2_lo <= add_tier1["high"]:
        t2_lo = _RoundPrice(add_tier1["high"] + 0.01)
    if t2_hi >= current_price:
        t2_hi = _RoundPrice(current_price - 0.01)
    if t2_hi <= t2_lo:
        t2_lo = _RoundPrice(add_tier1["high"] + 0.01)
        t2_hi = _RoundPrice(min(current_price - 0.01, t2_lo + 0.3))

    add_tier2 = _BuildTier(
        t2_lo,
        t2_hi,
        label="第二加仓（次选）",
        logic="昨日VWAP/MA5资金成本线，浅回调分歧位",
        rules=[
            "仅早盘快速回落、板块未全线跳水时少量加",
            "距现价空间有限，摊薄成本效果一般，不建议重仓",
        ],
        stop_loss=add_tier1.get("stop_loss", 0),
        anchor=t2_center,
        recommended=t2_center,
        recommended_source="rule",
    )

    # 卖出档（做T降本）
    s1_lo, s1_hi = _NarrowBand(pressure_t1, pct=0.004, min_half=0.2)
    if s1_lo <= current_price:
        s1_lo = _RoundPrice(current_price + 0.01)
    sell_tier1 = _BuildTier(
        s1_lo,
        max(s1_hi, s1_lo + 0.2),
        label="第一卖出（做T先减）",
        logic="反弹至今日高位/近端压力，先卖出加仓部分",
        rules=["反弹至此区间必须卖出加仓仓位，不留长线"],
        anchor=pressure_t1,
        recommended=pressure_t1,
        recommended_source="rule",
    )

    s2_lo, s2_hi = _NarrowBand(pressure_t2, pct=0.004, min_half=0.2)
    if s2_lo <= sell_tier1["high"]:
        s2_lo = _RoundPrice(sell_tier1["high"] + 0.01)
    sell_tier2 = _BuildTier(
        s2_lo,
        max(s2_hi, s2_lo + 0.2),
        label="第二卖出（做T再减）",
        logic="冲近5日高/远端压力，继续减仓",
        rules=["冲至此区间可继续卖出剩余加仓部分"],
        anchor=pressure_t2,
        recommended=pressure_t2,
        recommended_source="rule",
    )

    no_add_zones: list[dict[str, Any]] = []
    if forbidden > 0 and forbidden < current_price:
        no_add_zones.append({
            "low": _RoundPrice(forbidden),
            "high": 99999.0,
            "reason": "现价附近及上方属高位，短线获利盘随时兑现，追高加仓极易被套",
        })
    if key_support > 0:
        no_add_zones.append({
            "low": 0.0,
            "high": _RoundPrice(key_support),
            "reason": (
                f"有效跌破{key_support:.2f}则趋势转弱，"
                f"等{next_support:.2f}附近再评估，中途不加仓"
                if next_support > 0 else
                f"有效跌破{key_support:.2f}则趋势转弱，中途不加仓"
            ),
        })

    weighted = _SafeFloat((analysis or {}).get("weighted_score"))
    news_score = _SafeFloat((analysis or {}).get("news_score"))
    discipline = [
        "任何价位均禁止大仓位加仓，单次小仓位试探，仅为做T降本",
        f"反弹至{sell_tier1['low']:.2f}~{sell_tier2['high']:.2f}压力区必须卖出加仓部分",
        "若出现重大利空、低开低走，放弃所有加仓计划，先减仓避险",
    ]
    if news_score < -30:
        discipline.insert(0, "资讯面偏空，优先控制仓位，谨慎加仓")

    breakdown_plan = {
        "level": _RoundPrice(key_support),
        "next_support": _RoundPrice(next_support),
        "action": "有效跌破关键支撑则趋势转弱，不加仓，等待下一档支撑",
    }

    plan = {
        "risk_warning": _DEFAULT_RISK,
        "add_tier1": add_tier1,
        "add_tier2": add_tier2,
        "sell_tier1": sell_tier1,
        "sell_tier2": sell_tier2,
        "no_add_zones": no_add_zones,
        "discipline": discipline,
        "breakdown_plan": breakdown_plan,
    }
    return plan


def BuildRuleTradeZones(
    price_ctx: dict[str, Any],
    sr: dict[str, float],
    current_price: float,
    analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """兼容旧接口：从 trade_plan 提取四档区间。"""
    plan = BuildRuleTradePlan(price_ctx, sr, current_price, analysis)
    return {
        "add_tier1": plan["add_tier1"],
        "add_tier2": plan["add_tier2"],
        "sell_tier1": plan["sell_tier1"],
        "sell_tier2": plan["sell_tier2"],
    }


def _MergeTierFromLlm(
    llm_result: dict[str, Any],
    rule_tier: dict[str, Any],
    low_key: str,
    high_key: str,
    note_key: str,
    logic_key: str,
    rules_key: str,
    stop_key: str,
    current_price: float,
    side: Side,
    best_key: str = "",
) -> dict[str, Any]:
    raw_low = _SafeFloat(llm_result.get(low_key))
    raw_high = _SafeFloat(llm_result.get(high_key))
    logic = str(llm_result.get(logic_key, rule_tier.get("logic", ""))).strip()
    note = str(llm_result.get(note_key, rule_tier.get("note", ""))).strip()
    rules_raw = llm_result.get(rules_key)
    rules: list[str] = rule_tier.get("rules", [])
    if isinstance(rules_raw, list):
        rules = [str(r).strip() for r in rules_raw if str(r).strip()][:4] or rules
    stop_loss = _SafeFloat(llm_result.get(stop_key), rule_tier.get("stop_loss", 0))
    best = _SafeFloat(llm_result.get(best_key)) if best_key else 0.0

    valid_fn = _IsBuyZoneValid if side == "buy" else _IsSellZoneValid
    tier: dict[str, Any]
    if valid_fn(raw_low, raw_high, current_price):
        tier = _BuildTier(
            raw_low, raw_high,
            label=rule_tier.get("label", ""),
            logic=logic or rule_tier.get("logic", ""),
            rules=rules,
            stop_loss=stop_loss,
            note=note,
            anchor=_SafeFloat(rule_tier.get("anchor")),
        )
    elif side == "buy" and raw_low > 0 and raw_low < current_price:
        lo, hi = _NarrowBand(raw_low)
        if valid_fn(lo, hi, current_price):
            tier = _BuildTier(
                lo, hi,
                label=rule_tier.get("label", ""),
                logic=logic, rules=rules, stop_loss=stop_loss, note=note,
                anchor=_SafeFloat(rule_tier.get("anchor")),
            )
        else:
            tier = dict(rule_tier)
    elif side == "sell" and raw_high > 0 and raw_high > current_price:
        lo, hi = _NarrowBand(raw_high)
        if valid_fn(lo, hi, current_price):
            tier = _BuildTier(
                lo, hi,
                label=rule_tier.get("label", ""),
                logic=logic, rules=rules, note=note,
                anchor=_SafeFloat(rule_tier.get("anchor")),
            )
        else:
            tier = dict(rule_tier)
    else:
        tier = dict(rule_tier)

    lo = _SafeFloat(tier.get("low"))
    hi = _SafeFloat(tier.get("high"))
    if best > 0 and lo <= best <= hi:
        tier["recommended"] = _RoundPrice(best)
        tier["recommended_source"] = "ai"
    elif not _SafeFloat(tier.get("recommended")):
        tier["recommended"] = ResolveTierRecommended(tier, side)
        tier["recommended_source"] = str(
            tier.get("recommended_source") or rule_tier.get("recommended_source") or "rule"
        )
    if not _SafeFloat(tier.get("anchor")):
        tier["anchor"] = _SafeFloat(rule_tier.get("anchor"))
    return tier


def _ParseNoAddZones(
    llm_result: dict[str, Any],
    rule_zones: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    raw = llm_result.get("no_add_zones")
    if not isinstance(raw, list) or not raw:
        return rule_zones
    parsed: list[dict[str, Any]] = []
    for item in raw[:4]:
        if not isinstance(item, dict):
            continue
        reason = str(item.get("reason", "")).strip()
        if not reason:
            continue
        parsed.append({
            "low": _RoundPrice(_SafeFloat(item.get("low"))),
            "high": _RoundPrice(_SafeFloat(item.get("high"), 99999.0)),
            "reason": reason[:120],
        })
    return parsed or rule_zones


def ParseTradePlanFromLlm(
    llm_result: dict[str, Any] | None,
    rule_plan: dict[str, Any],
    current_price: float,
) -> dict[str, Any]:
    """解析 AI 完整交易计划，价格/逻辑 LLM 优先，无效时回退规则。"""
    if not llm_result or current_price <= 0:
        return rule_plan

    plan = dict(rule_plan)
    plan["add_tier1"] = _MergeTierFromLlm(
        llm_result, rule_plan["add_tier1"],
        "add_tier1_low", "add_tier1_high", "add_tier1_note",
        "add_tier1_logic", "add_tier1_rules", "add_tier1_stop",
        current_price, "buy", "add_tier1_best",
    )
    plan["add_tier2"] = _MergeTierFromLlm(
        llm_result, rule_plan["add_tier2"],
        "add_tier2_low", "add_tier2_high", "add_tier2_note",
        "add_tier2_logic", "add_tier2_rules", "add_tier2_stop",
        current_price, "buy", "add_tier2_best",
    )
    plan["sell_tier1"] = _MergeTierFromLlm(
        llm_result, rule_plan["sell_tier1"],
        "sell_tier1_low", "sell_tier1_high", "sell_tier1_note",
        "sell_tier1_logic", "sell_tier1_rules", "sell_tier1_stop",
        current_price, "sell", "sell_tier1_best",
    )
    plan["sell_tier2"] = _MergeTierFromLlm(
        llm_result, rule_plan["sell_tier2"],
        "sell_tier2_low", "sell_tier2_high", "sell_tier2_note",
        "sell_tier2_logic", "sell_tier2_rules", "sell_tier2_stop",
        current_price, "sell", "sell_tier2_best",
    )

    t1, t2 = plan["add_tier1"], plan["add_tier2"]
    if t1["high"] > 0 and t2["low"] > 0 and t2["low"] <= t1["high"]:
        t2["low"] = _RoundPrice(t1["high"] + 0.01)
        if t2["high"] <= t2["low"]:
            t2["high"] = _RoundPrice(t2["low"] + 0.3)

    s1, s2 = plan["sell_tier1"], plan["sell_tier2"]
    if s1["high"] > 0 and s2["low"] > 0 and s2["low"] <= s1["high"]:
        s2["low"] = _RoundPrice(s1["high"] + 0.01)
        if s2["high"] <= s2["low"]:
            s2["high"] = _RoundPrice(s2["low"] + 0.3)

    risk = str(llm_result.get("risk_warning", "")).strip()
    if risk:
        plan["risk_warning"] = risk[:200]

    plan["no_add_zones"] = _ParseNoAddZones(
        llm_result, rule_plan.get("no_add_zones", []),
    )

    discipline_raw = llm_result.get("trade_discipline")
    if isinstance(discipline_raw, list):
        disc = [str(d).strip() for d in discipline_raw if str(d).strip()][:5]
        if disc:
            plan["discipline"] = disc

    bd_level = _SafeFloat(llm_result.get("breakdown_level"))
    bd_next = _SafeFloat(llm_result.get("breakdown_next_support"))
    bd_action = str(llm_result.get("breakdown_action", "")).strip()
    if bd_level > 0 or bd_action:
        plan["breakdown_plan"] = {
            "level": _RoundPrice(bd_level or rule_plan["breakdown_plan"]["level"]),
            "next_support": _RoundPrice(
                bd_next or rule_plan["breakdown_plan"]["next_support"],
            ),
            "action": bd_action or rule_plan["breakdown_plan"]["action"],
        }

    return plan


def ParseTradeZonesFromLlm(
    llm_result: dict[str, Any] | None,
    rule_zones: dict[str, Any],
    current_price: float,
) -> dict[str, Any]:
    """兼容旧接口：从 trade_plan 解析四档。"""
    rule_plan = {
        "risk_warning": _DEFAULT_RISK,
        "add_tier1": rule_zones.get("add_tier1", _EmptyTier()),
        "add_tier2": rule_zones.get("add_tier2", _EmptyTier()),
        "sell_tier1": rule_zones.get("sell_tier1", _EmptyTier()),
        "sell_tier2": rule_zones.get("sell_tier2", _EmptyTier()),
        "no_add_zones": [],
        "discipline": [],
        "breakdown_plan": {"level": 0.0, "next_support": 0.0, "action": ""},
    }
    plan = ParseTradePlanFromLlm(llm_result, rule_plan, current_price)
    return {
        "add_tier1": plan["add_tier1"],
        "add_tier2": plan["add_tier2"],
        "sell_tier1": plan["sell_tier1"],
        "sell_tier2": plan["sell_tier2"],
    }


def EnvelopeFromTradeZones(zones: dict[str, Any]) -> dict[str, float]:
    """从四档区间推导买卖总包络（兼容旧字段）。"""
    t1 = zones.get("add_tier1", {})
    t2 = zones.get("add_tier2", {})
    s1 = zones.get("sell_tier1", {})
    s2 = zones.get("sell_tier2", {})
    buy_low = _SafeFloat(t1.get("low"))
    buy_high = _SafeFloat(t2.get("high"))
    sell_low = _SafeFloat(s1.get("low"))
    sell_high = _SafeFloat(s2.get("high"))
    if buy_low > buy_high and buy_high > 0:
        buy_low, buy_high = buy_high, buy_low
    if sell_low > sell_high and sell_high > 0:
        sell_low, sell_high = sell_high, sell_low
    return {
        "buy_low": buy_low,
        "buy_high": buy_high,
        "sell_low": sell_low,
        "sell_high": sell_high,
    }
