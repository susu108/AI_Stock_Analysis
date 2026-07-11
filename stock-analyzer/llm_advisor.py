"""DeepSeek 大模型分析模块 — 日常分析 / 持仓建议。"""

from __future__ import annotations

import json
import re
from typing import Any

import requests

import config
from market_context import ResolvePredictionHorizon
from portfolio import CalcPortfolioSummary, FormatPortfolioText, PortfolioToDict
from price_context import FormatPriceContextText
from trade_zones import EnvelopeFromTradeZones, ParseTradePlanFromLlm
from utils import SetupLogger

logger = SetupLogger(config.LOG_LEVEL)

_LLM_TIMEOUT = 90
_VALID_DIRECTIONS = frozenset({"看涨", "看跌", "震荡偏涨", "震荡偏跌", "震荡"})


def IsLlmEnabled() -> bool:
    """判断是否启用大模型分析。"""
    if not config.LLM_ENABLED:
        return False
    if not config.DEEPSEEK_API_KEY:
        logger.warning("未配置 DEEPSEEK_API_KEY，跳过大模型分析")
        return False
    return True


def _FormatNewsSection(
    title: str,
    items: list[dict[str, Any]],
    max_count: int = 6,
) -> list[str]:
    """格式化某一类资讯段落。"""
    if not items:
        return []
    lines = [f"【{title}】"]
    for i, item in enumerate(items[:max_count], 1):
        impact_tag = f"[{item.get('impact', '')}]" if item.get("impact") else ""
        lines.append(
            f"{i}. {impact_tag}[{item.get('time', '')}] {item.get('title', '')} "
            f"（{item.get('source', '')}）"
        )
        if item.get("impact_reason"):
            lines.append(f"   影响：{item['impact_reason']}")
        elif item.get("content"):
            lines.append(f"   摘要：{item['content']}")
    return lines


def BuildAnalysisContext(
    analysis: dict[str, Any],
    data: dict[str, Any],
    base_advice: dict[str, Any],
    price_ctx: dict[str, Any],
    news_items: list[dict[str, str]],
    session_label: str,
    news_bundle: dict[str, Any] | None = None,
    mode: str = "daily",
) -> str:
    """构建提交给大模型的分析上下文。"""
    rt = data.get("realtime") or {}
    indicators = analysis.get("indicators", {})
    current_price = float(rt.get("price", indicators.get("price", 0)))

    context_parts = [
        f"股票：{config.STOCK_NAME}({config.STOCK_CODE})",
        f"推送时段：{session_label}",
    ]
    horizon = base_advice.get("prediction_horizon") or ResolvePredictionHorizon(session_label)
    context_parts.extend([
        "",
        "【预测Horizon】",
        f"- 当前时段：{horizon.get('session_label', session_label)}",
        f"- 近端预测目标：{horizon.get('near_term_target', '')}（{horizon.get('near_term_label', '')}）",
        f"- 市场是否开盘：{'是' if horizon.get('is_market_open') else '否'}",
        "- 要求：近端/短期/长期三层预测均须综合K线技术面信号与全部有影响资讯",
        "",
        "【实时行情】",
        f"- 最新价：{current_price}",
        f"- 涨跌幅：{rt.get('change_pct', 0)}%",
        f"- 换手率：{rt.get('turnover', 0)}%",
        f"- 量比：{rt.get('volume_ratio', 0)}",
        f"- 市盈率：{rt.get('pe', 0)}",
        f"- 市净率：{rt.get('pb', 0)}",
        "",
        "【综合预测】",
        f"- 方向：{analysis.get('direction', '')} {analysis.get('direction_icon', '')}",
        f"- 综合评分：{analysis.get('weighted_score', 0):+.1f}",
        f"- 预测涨跌区间：{analysis.get('change_low', 0):+.2f}% ~ {analysis.get('change_high', 0):+.2f}%",
        f"- 信心指数：{analysis.get('confidence', 1)}/5",
        f"- 技术面评分：{analysis.get('tech_score', 0):+.0f}（权重35%）",
        f"- 资金面评分：{analysis.get('fund_score', 0):+.0f}（权重30%）",
        f"- 板块面评分：{analysis.get('sector_score', 0):+.0f}（权重20%）",
        f"- 资讯面评分：{analysis.get('news_score', 0):+.0f}（权重15%）",
        "",
        "【技术面信号】",
        *[f"- {s}" for s in analysis.get("tech_signals", [])],
        "",
        "【资金面信号】",
        *[f"- {s}" for s in analysis.get("fund_signals", [])],
        "",
        "【板块面信号】",
        *[f"- {s}" for s in analysis.get("sector_signals", [])],
    ])
    news_signals = analysis.get("news_signals", [])
    if news_signals:
        context_parts.extend([
            "",
            "【资讯面信号】",
            *[f"- {s}" for s in news_signals[:6]],
        ])
    context_parts.extend([
        "",
        "【历史价位参考】",
        *FormatPriceContextText(price_ctx, include_cost_lines=(mode != "daily")),
        "",
        "【量化支撑/压力位（K线参考，定价须结合资讯与技术面综合判定）】",
        f"- 第一支撑位 S1：{base_advice.get('s1', 0)}",
        f"- 第二支撑位 S2：{base_advice.get('s2', 0)}",
        f"- 止损位：{base_advice.get('stop_loss', 0)}",
        f"- 第一压力位 R1：{base_advice.get('r1', 0)}",
        f"- 第二压力位 R2：{base_advice.get('r2', 0)}",
    ])
    plan = base_advice.get("trade_plan") or {}
    zones = plan if plan.get("add_tier1") else (base_advice.get("trade_zones") or {})
    if zones:
        at1 = zones.get("add_tier1", {})
        at2 = zones.get("add_tier2", {})
        st1 = zones.get("sell_tier1", {})
        st2 = zones.get("sell_tier2", {})
        context_parts.extend([
            "",
            "【分档区间参考（规则引擎，AI 须结合技术锚点输出更精准窄区间）】",
            f"- 第一加仓（深支撑/优先等）：{at1.get('low', 0)} ~ {at1.get('high', 0)}"
            f"（推荐 {at1.get('recommended', 0)}）",
            f"- 第二加仓（浅回调/次选）：{at2.get('low', 0)} ~ {at2.get('high', 0)}"
            f"（推荐 {at2.get('recommended', 0)}）",
            f"- 第一卖出（做T先减）：{st1.get('low', 0)} ~ {st1.get('high', 0)}",
            f"- 第二卖出（做T再减）：{st2.get('low', 0)} ~ {st2.get('high', 0)}",
        ])
        if plan.get("no_add_zones"):
            context_parts.append("- 禁加区（规则）：")
            for z in plan["no_add_zones"][:3]:
                lo = z.get("low", 0)
                hi = z.get("high", 0)
                context_parts.append(f"  · {lo}~{hi}：{z.get('reason', '')}")
    context_parts.extend([
        "",
        "【AI定价硬性约束 — 输出 JSON 时必须满足】",
        f"- 现价：{current_price}",
        "- 四档均含参考区间（每档宽度约0.3~1.2元）及 _best 最推荐单点价；须结合箱体/VWAP/关键支撑/今日高",
        "- add_tier1 为深支撑（低于 add_tier2）；add_tier2 为浅回调次选",
        "- sell_tier2 高于 sell_tier1；须给出 risk_warning、no_add_zones、trade_discipline",
        f"- add_tier2_high < {current_price} < sell_tier1_low",
        "- buy_price_low=add_tier1_low，buy_price_high=add_tier2_high（四档总包络）",
    ])

    if mode != "daily":
        portfolio = CalcPortfolioSummary(current_price)
        if portfolio is not None:
            context_parts.extend([
                "",
                "【我的持仓】",
                *FormatPortfolioText(portfolio, current_price),
            ])

    bundle = news_bundle or {}
    if bundle.get("sector_snapshot"):
        context_parts.extend(["", "【板块涨跌快照（影响个股走势）】"])
        context_parts.extend(bundle["sector_snapshot"])

    fetched_at = bundle.get("fetched_at", "")
    if fetched_at:
        context_parts.extend([
            "",
            f"【资讯说明】以下为本次实时采集（{fetched_at}），"
            "涨跌预判与买卖建议必须结合有影响资讯综合分析。",
        ])

    relevant_news = bundle.get("relevant_items") or news_items
    if relevant_news:
        context_parts.extend(["", "【有影响资讯 — 涨跌与买卖建议必须结合以下信息】"])
        by_cat: dict[str, list[dict[str, Any]]] = {}
        for item in relevant_news:
            cat = item.get("category", "stock")
            by_cat.setdefault(cat, []).append(item)
        cat_labels = {
            "stock": "个股资讯",
            "sector": "板块/行业",
            "policy": "政策/监管",
            "macro": "宏观事件",
        }
        for cat, label in cat_labels.items():
            if by_cat.get(cat):
                context_parts.extend([""] + _FormatNewsSection(label, by_cat[cat], 6))

    return "\n".join(context_parts)


def _BuildSystemPrompt(mode: str = "daily") -> str:
    stock = config.STOCK_CODE
    return (
        "你是一位专业的 A 股分析师，擅长结合技术面、资金面、板块面、政策面、宏观面"
        "与多源资讯给出实用交易建议。"
        "请基于用户提供的数据进行分析，输出严格的 JSON 格式，不要包含 markdown 代码块。"
        "JSON 字段说明："
        "strategy（字符串，150字以内，操作策略须提及资讯/政策/板块影响）；"
        "buy_action（枚举：买入/轻仓买入/观望）；"
        "sell_action（枚举：卖出/减仓/持有）；"
        "buy_reasons（字符串数组，2-4条，每条35字以内，至少1条必须引用资讯/政策/板块）；"
        "sell_reasons（字符串数组，1-4条，每条35字以内，至少1条必须引用资讯/政策/板块）；"
        "buy_price_low（数字，第一加仓下限=add_tier1_low，深支撑）；"
        "buy_price_high（数字，第二加仓上限=add_tier2_high，须低于现价）；"
        "sell_price_low（数字，第一卖出下限=sell_tier1_low，须高于现价）；"
        "sell_price_high（数字，第二卖出上限=sell_tier2_high）；"
        "risk_warning（字符串，80字，题材/基本面风险提示，不构成投资建议）；"
        "add_tier1_low/add_tier1_high（数字，第一加仓参考区间，宽度宜0.3~1.2元）；"
        "add_tier1_best（数字，第一加仓最推荐单点价，须在 low~high 内，优先箱体中枢/深支撑承接位）；"
        "add_tier1_logic（字符串，60字，第一加仓技术逻辑）；"
        "add_tier1_rules（字符串数组，2-3条操作规则，含仓位/触发条件）；"
        "add_tier1_stop（数字，第一加仓止损价）；"
        "add_tier1_note（字符串，40字以内，兼容字段）；"
        "add_tier2_low/add_tier2_high（数字，第二加仓参考区间，VWAP/MA5浅回调，次选）；"
        "add_tier2_best（数字，第二加仓最推荐单点价，须在 low~high 内）；"
        "add_tier2_logic/add_tier2_rules/add_tier2_stop/add_tier2_note（同上结构）；"
        "sell_tier1_low/sell_tier1_high（数字，第一卖出参考区间，今日高/近端压力，做T先减）；"
        "sell_tier1_best（数字，第一卖出最推荐单点价，须在 low~high 内）；"
        "sell_tier1_logic/sell_tier1_rules/sell_tier1_note（同上结构）；"
        "sell_tier2_low/sell_tier2_high（数字，第二卖出参考区间，近5日高，做T再减）；"
        "sell_tier2_best（数字，第二卖出最推荐单点价，须在 low~high 内）；"
        "sell_tier2_logic/sell_tier2_rules/sell_tier2_note（同上结构）；"
        "no_add_zones（数组，每项{low,high,reason}，不宜加仓的价位区间）；"
        "trade_discipline（字符串数组，3-5条加仓/做T硬性纪律）；"
        "breakdown_level/breakdown_next_support/breakdown_action（跌破关键支撑后的应对）；"
        "pricing_rationale（字符串，80字以内，说明四档如何结合K线、资讯确定）；"
        "news_summary（字符串，150字以内）；"
        "sector_impact（字符串，80字以内）；"
        "policy_impact（字符串，120字以内，政策类资讯须完整提炼要点）；"
        "news_impact（字符串，80字以内）；"
        "near_term_direction（枚举：看涨/看跌/震荡偏涨/震荡偏跌/震荡）；"
        "near_term_change_low（数字，近端涨跌幅下限%，如-1.5）；"
        "near_term_change_high（数字，近端涨跌幅上限%，如2.0）；"
        "near_term_prediction（字符串，80字以内，综合K线+资讯的近端研判）；"
        "near_term_advice（字符串，60字以内，近端操作建议）；"
        "short_term_direction（枚举：看涨/看跌/震荡偏涨/震荡偏跌/震荡，短期1~2周）；"
        "short_term_prediction（字符串，80字以内）；"
        "short_term_advice（字符串，60字以内，短期操作建议）；"
        "long_term_direction（枚举：看涨/看跌/震荡偏涨/震荡偏跌/震荡，长期1~3月）；"
        "long_term_prediction（字符串，80字以内）；"
        "long_term_advice（字符串，60字以内，长期操作建议）；"
        "direction_prediction（字符串，80字以内，与near_term_prediction一致，兼容字段）；"
        "news_conclusion（字符串，150字以内，资讯综合建议：置于资讯解读末尾，"
        f"须综合全部利好/利空/政策/板块资讯，给出对{stock}短期操作的明确结论与注意事项）。"
        "分析原则："
        "1. 近端/短期/长期三层预测与买卖建议必须综合K线信号与有影响资讯；"
        "2. 开盘前/收盘后：near_term 针对下一交易日；盘中：near_term 针对今日剩余时段至收盘；"
        "3. 第一加仓=深支撑/箱体中枢（优先等待），第二加仓=浅回调/VWAP（次选，限制更严）；"
        "4. 须输出 risk_warning、四档参考区间及 logic/rules/stop、每档 _best 最推荐单点价、no_add_zones、trade_discipline；"
        "5. buy/sell 四价为包络：buy_price_low=add_tier1_low，buy_price_high=add_tier2_high；"
        "6. 定价须优先参考【技术锚点】箱体/VWAP/关键支撑/禁加线/做T压力；"
        "7. 第一加仓 best 应贴近箱体中枢/深支撑承接位，勿简单取区间中点；区间过宽时 best 为实际操作价；"
        "8. 结合资讯判断题材属性与利空场景（如利空低开放弃加仓）；股数建议写在 rules 中；"
        "9. 理由必须引用提供的数据，不要编造；"
        "10. 仅输出 JSON，不要有其他文字。"
    )


def CallDeepSeek(context: str, mode: str = "daily") -> dict[str, Any] | None:
    """调用 DeepSeek Chat API，返回解析后的 JSON 字典。"""
    url = f"{config.DEEPSEEK_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": _BuildSystemPrompt(mode)},
            {"role": "user", "content": context},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.3,
        "max_tokens": 2048,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=_LLM_TIMEOUT)
        resp.raise_for_status()
        body = resp.json()
        content = body["choices"][0]["message"]["content"]
        return _ParseLlmJson(content)
    except requests.Timeout:
        logger.error("DeepSeek API 请求超时（%ds）", _LLM_TIMEOUT)
        return None
    except requests.HTTPError as exc:
        logger.error(
            "DeepSeek API HTTP 错误: %s — %s",
            exc,
            exc.response.text[:200] if exc.response else "",
        )
        return None
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        logger.error("DeepSeek 响应解析失败: %s", exc)
        return None
    except Exception as exc:
        logger.error("DeepSeek 调用异常: %s", exc)
        return None


def _ParseLlmJson(content: str) -> dict[str, Any] | None:
    """解析大模型返回的 JSON 内容。"""
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        result = json.loads(text)
        if not isinstance(result, dict):
            logger.error("DeepSeek 返回非 dict 类型")
            return None
        return result
    except json.JSONDecodeError as exc:
        logger.error("JSON 解析失败: %s — 原文: %s", exc, text[:200])
        return None


def _NormalizeReasons(raw: Any, max_count: int = 4) -> list[str]:
    """规范化理由列表。"""
    if not isinstance(raw, list):
        return []
    reasons: list[str] = []
    for item in raw:
        text = str(item).strip()
        if text:
            reasons.append(text)
        if len(reasons) >= max_count:
            break
    return reasons


def _SafeFloat(value: Any, default: float = 0.0) -> float:
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return default


def _MapBuyOk(buy_action: str) -> bool:
    return buy_action in ("买入", "轻仓买入")


def _MapSellOk(sell_action: str) -> bool:
    return sell_action in ("卖出", "减仓")


def _IsPriceRangeValid(
    buy_low: float,
    buy_high: float,
    sell_low: float,
    sell_high: float,
    current_price: float,
) -> bool:
    """判断买卖四价是否满足基本约束。"""
    if current_price <= 0:
        return False
    return (
        buy_low > 0
        and buy_high > 0
        and sell_low > 0
        and sell_high > 0
        and buy_low <= buy_high < current_price < sell_low <= sell_high
    )


def _RepairAiPrices(
    buy_low: float,
    buy_high: float,
    sell_low: float,
    sell_high: float,
    current_price: float,
    rule_prices: dict[str, float],
) -> dict[str, float]:
    """在 AI 定价不完全合法时，尽量保留 AI 意图并修正为可执行区间。"""
    s1 = float(rule_prices.get("buy_high", rule_prices.get("rule_s1", 0)) or 0)
    s2 = float(rule_prices.get("buy_low", rule_prices.get("rule_s2", 0)) or 0)
    r1 = float(rule_prices.get("sell_low", rule_prices.get("rule_r1", 0)) or 0)
    r2 = float(rule_prices.get("sell_high", rule_prices.get("rule_r2", 0)) or 0)

    if buy_low <= 0 and s2 > 0:
        buy_low = s2
    if buy_high <= 0 and s1 > 0:
        buy_high = s1
    if sell_low <= 0 and r1 > 0:
        sell_low = r1
    if sell_high <= 0 and r2 > 0:
        sell_high = r2

    if buy_low > buy_high and buy_high > 0:
        buy_low, buy_high = buy_high, buy_low
    if sell_low > sell_high and sell_high > 0:
        sell_low, sell_high = sell_high, sell_low

    max_buy_high = round(current_price - 0.01, 2)
    if buy_high > max_buy_high:
        buy_high = max_buy_high
    if buy_low > buy_high:
        buy_low = round(buy_high * 0.97, 2)

    min_sell_low = round(current_price + 0.01, 2)
    if sell_low < min_sell_low:
        sell_low = min_sell_low
    if sell_high < sell_low:
        sell_high = round(sell_low * 1.03, 2)

    return {
        "buy_low": round(buy_low, 2),
        "buy_high": round(buy_high, 2),
        "sell_low": round(sell_low, 2),
        "sell_high": round(sell_high, 2),
    }


def ResolveAiPrices(
    llm_result: dict[str, Any],
    current_price: float,
    rule_prices: dict[str, float],
) -> tuple[dict[str, float], str]:
    """解析 AI 定价，优先采用 AI 输出，必要时智能修正。"""
    raw_buy_low = _SafeFloat(llm_result.get("buy_price_low"))
    raw_buy_high = _SafeFloat(llm_result.get("buy_price_high"))
    raw_sell_low = _SafeFloat(llm_result.get("sell_price_low"))
    raw_sell_high = _SafeFloat(llm_result.get("sell_price_high"))

    if _IsPriceRangeValid(
        raw_buy_low, raw_buy_high, raw_sell_low, raw_sell_high, current_price,
    ):
        return {
            "buy_low": raw_buy_low,
            "buy_high": raw_buy_high,
            "sell_low": raw_sell_low,
            "sell_high": raw_sell_high,
        }, "ai"

    repaired = _RepairAiPrices(
        raw_buy_low, raw_buy_high, raw_sell_low, raw_sell_high,
        current_price, rule_prices,
    )
    if _IsPriceRangeValid(
        repaired["buy_low"], repaired["buy_high"],
        repaired["sell_low"], repaired["sell_high"],
        current_price,
    ):
        logger.warning(
            "AI 定价已修正 — 原始:%.2f~%.2f / %.2f~%.2f → %.2f~%.2f / %.2f~%.2f",
            raw_buy_low, raw_buy_high, raw_sell_low, raw_sell_high,
            repaired["buy_low"], repaired["buy_high"],
            repaired["sell_low"], repaired["sell_high"],
        )
        return repaired, "ai_repaired"

    fallback = {
        "buy_low": float(rule_prices.get("buy_low", 0)),
        "buy_high": float(rule_prices.get("buy_high", 0)),
        "sell_low": float(rule_prices.get("sell_low", 0)),
        "sell_high": float(rule_prices.get("sell_high", 0)),
    }
    logger.warning(
        "AI 定价无法修正，回退 K 线规则 — 原始:%.2f~%.2f / %.2f~%.2f",
        raw_buy_low, raw_buy_high, raw_sell_low, raw_sell_high,
    )
    return fallback, "rule_fallback"


def _ValidatePriceRange(
    buy_low: float,
    buy_high: float,
    sell_low: float,
    sell_high: float,
    current_price: float,
    fallback: dict[str, float],
) -> dict[str, float] | None:
    """校验 AI 定价是否满足基本约束，无效时返回 None。"""
    if _IsPriceRangeValid(buy_low, buy_high, sell_low, sell_high, current_price):
        return {
            "buy_low": buy_low,
            "buy_high": buy_high,
            "sell_low": sell_low,
            "sell_high": sell_high,
        }
    return None


def _NormalizeDirection(raw: Any, fallback: str = "震荡") -> str:
    """规范化方向枚举。"""
    direction = str(raw or fallback).strip()
    if direction in _VALID_DIRECTIONS:
        return direction
    return fallback if fallback in _VALID_DIRECTIONS else "震荡"


def BuildRuleFallbackPredictions(
    analysis: dict[str, Any],
    horizon: dict[str, Any],
    base_advice: dict[str, Any],
) -> dict[str, Any]:
    """LLM 不可用时，基于量化分析生成三层预测降级文案。"""
    direction = str(analysis.get("direction", "震荡"))
    change_low = float(analysis.get("change_low", 0))
    change_high = float(analysis.get("change_high", 0))
    tech_hint = ""
    tech_signals = analysis.get("tech_signals", [])
    if tech_signals:
        tech_hint = tech_signals[0]
    news_hint = ""
    news_signals = analysis.get("news_signals", [])
    if news_signals:
        news_hint = news_signals[0]

    near_label = str(horizon.get("near_term_label", "近端预测"))
    target = str(horizon.get("near_term_target", ""))
    strategy = str(base_advice.get("strategy", "")).strip()
    near_prediction = (
        f"量化综合{direction}（{change_low:+.1f}%~{change_high:+.1f}%），"
        f"针对{target}。"
    )
    if tech_hint:
        near_prediction += f" K线：{tech_hint[:40]}。"
    if news_hint:
        near_prediction += f" 资讯：{news_hint[:40]}。"

    near_advice = strategy or "建议观望，等待方向明确后再操作"
    short_prediction = (
        f"短期1~2周参考量化方向{direction}，"
        f"技术{analysis.get('tech_score', 0):+.0f}资金{analysis.get('fund_score', 0):+.0f}。"
    )
    long_prediction = (
        f"长期1~3月需关注板块与政策面，"
        f"当前四维综合{analysis.get('weighted_score', 0):+.1f}。"
    )

    return {
        "near_term": {
            "label": near_label,
            "direction": _NormalizeDirection(direction),
            "change_low": change_low,
            "change_high": change_high,
            "prediction": near_prediction[:120],
            "advice": near_advice[:80],
        },
        "short_term": {
            "label": "短期（1~2周）",
            "direction": _NormalizeDirection(direction),
            "prediction": short_prediction[:120],
            "advice": "波段操作宜控制仓位，结合资讯与技术面动态调整"[:80],
        },
        "long_term": {
            "label": "长期（1~3月）",
            "direction": _NormalizeDirection(direction),
            "prediction": long_prediction[:120],
            "advice": "长期布局需关注基本面与政策导向，不宜追涨杀跌"[:80],
        },
        "source": "rule_fallback",
    }


def ParsePredictionsFromLlm(
    llm_result: dict[str, Any],
    horizon: dict[str, Any],
    analysis: dict[str, Any],
) -> dict[str, Any]:
    """从 LLM 结果解析三层预测结构。"""
    near_prediction = str(
        llm_result.get("near_term_prediction")
        or llm_result.get("direction_prediction", "")
    ).strip()
    default_dir = str(analysis.get("direction", "震荡"))
    near_change_low = _SafeFloat(
        llm_result.get("near_term_change_low"),
        float(analysis.get("change_low", 0)),
    )
    near_change_high = _SafeFloat(
        llm_result.get("near_term_change_high"),
        float(analysis.get("change_high", 0)),
    )
    if near_change_low > near_change_high:
        near_change_low, near_change_high = near_change_high, near_change_low

    return {
        "near_term": {
            "label": str(horizon.get("near_term_label", "近端预测")),
            "direction": _NormalizeDirection(
                llm_result.get("near_term_direction"), default_dir,
            ),
            "change_low": near_change_low,
            "change_high": near_change_high,
            "prediction": near_prediction[:120],
            "advice": str(llm_result.get("near_term_advice", "")).strip()[:80],
        },
        "short_term": {
            "label": "短期（1~2周）",
            "direction": _NormalizeDirection(
                llm_result.get("short_term_direction"), default_dir,
            ),
            "prediction": str(llm_result.get("short_term_prediction", "")).strip()[:120],
            "advice": str(llm_result.get("short_term_advice", "")).strip()[:80],
        },
        "long_term": {
            "label": "长期（1~3月）",
            "direction": _NormalizeDirection(
                llm_result.get("long_term_direction"), default_dir,
            ),
            "prediction": str(llm_result.get("long_term_prediction", "")).strip()[:120],
            "advice": str(llm_result.get("long_term_advice", "")).strip()[:80],
        },
        "source": "ai",
    }


def EnhanceAdviceWithLlm(
    analysis: dict[str, Any],
    data: dict[str, Any],
    base_advice: dict[str, Any],
    price_ctx: dict[str, Any],
    news_items: list[dict[str, str]],
    session_label: str = "盘中",
    news_bundle: dict[str, Any] | None = None,
    mode: str = "daily",
) -> dict[str, Any]:
    """使用 DeepSeek 增强买卖建议，失败时返回原始建议。"""
    rt = data.get("realtime") or {}
    indicators = analysis.get("indicators", {})
    current_price = float(rt.get("price", indicators.get("price", 0)))

    base_advice["news_items"] = news_items
    base_advice["news_bundle"] = news_bundle or {}
    base_advice["session_label"] = session_label
    base_advice["price_context"] = price_ctx
    base_advice["report_mode"] = mode
    horizon = base_advice.get("prediction_horizon") or ResolvePredictionHorizon(session_label)
    base_advice["prediction_horizon"] = horizon

    if not IsLlmEnabled():
        base_advice["llm_used"] = False
        base_advice["predictions"] = BuildRuleFallbackPredictions(
            analysis, horizon, base_advice,
        )
        base_advice["direction_prediction"] = base_advice["predictions"]["near_term"]["prediction"]
        return base_advice

    context = BuildAnalysisContext(
        analysis, data, base_advice, price_ctx, news_items,
        session_label, news_bundle, mode,
    )
    logger.info("正在调用 DeepSeek 生成综合分析...")
    llm_result = CallDeepSeek(context, mode)

    if llm_result is None:
        logger.warning("DeepSeek 分析失败，使用规则引擎结果")
        base_advice["llm_used"] = False
        base_advice["predictions"] = BuildRuleFallbackPredictions(
            analysis, horizon, base_advice,
        )
        base_advice["direction_prediction"] = base_advice["predictions"]["near_term"]["prediction"]
        return base_advice

    strategy = str(llm_result.get("strategy", "")).strip()
    buy_action = str(llm_result.get("buy_action", "观望")).strip()
    sell_action = str(llm_result.get("sell_action", "持有")).strip()
    buy_reasons = _NormalizeReasons(llm_result.get("buy_reasons"))
    sell_reasons = _NormalizeReasons(llm_result.get("sell_reasons"))

    fallback_prices = {
        "buy_low": float(base_advice.get("buy_low", 0)),
        "buy_high": float(base_advice.get("buy_high", 0)),
        "sell_low": float(base_advice.get("sell_low", 0)),
        "sell_high": float(base_advice.get("sell_high", 0)),
        "rule_s1": float(base_advice.get("rule_s1", 0)),
        "rule_s2": float(base_advice.get("rule_s2", 0)),
        "rule_r1": float(base_advice.get("rule_r1", 0)),
        "rule_r2": float(base_advice.get("rule_r2", 0)),
    }
    rule_plan = base_advice.get("trade_plan") or {
        "risk_warning": "",
        "add_tier1": (base_advice.get("trade_zones") or {}).get("add_tier1", {}),
        "add_tier2": (base_advice.get("trade_zones") or {}).get("add_tier2", {}),
        "sell_tier1": (base_advice.get("trade_zones") or {}).get("sell_tier1", {}),
        "sell_tier2": (base_advice.get("trade_zones") or {}).get("sell_tier2", {}),
        "no_add_zones": [],
        "discipline": [],
        "breakdown_plan": {"level": 0.0, "next_support": 0.0, "action": ""},
    }
    trade_plan = ParseTradePlanFromLlm(llm_result, rule_plan, current_price)
    trade_zones = {
        "add_tier1": trade_plan["add_tier1"],
        "add_tier2": trade_plan["add_tier2"],
        "sell_tier1": trade_plan["sell_tier1"],
        "sell_tier2": trade_plan["sell_tier2"],
    }
    tier_envelope = EnvelopeFromTradeZones(trade_zones)
    fallback_prices.update(tier_envelope)

    final_prices, pricing_source = ResolveAiPrices(
        llm_result, current_price, fallback_prices,
    )

    if strategy:
        base_advice["strategy"] = strategy
    if buy_reasons:
        base_advice["buy_reasons"] = buy_reasons
    if sell_reasons:
        base_advice["sell_reasons"] = sell_reasons

    base_advice["trade_plan"] = trade_plan
    base_advice["trade_zones"] = trade_zones
    base_advice["buy_low"] = tier_envelope["buy_low"]
    base_advice["buy_high"] = tier_envelope["buy_high"]
    base_advice["sell_low"] = tier_envelope["sell_low"]
    base_advice["sell_high"] = tier_envelope["sell_high"]
    tier1_stop = float(trade_plan.get("add_tier1", {}).get("stop_loss", 0) or 0)
    if tier1_stop > 0:
        base_advice["stop_loss"] = tier1_stop
    elif tier_envelope["buy_low"] > 0:
        base_advice["stop_loss"] = round(tier_envelope["buy_low"] * 0.98, 2)
    base_advice["buy_ok"] = _MapBuyOk(buy_action)
    base_advice["sell_ok"] = _MapSellOk(sell_action)
    base_advice["buy_action"] = buy_action
    base_advice["sell_action"] = sell_action

    base_advice["news_summary"] = str(llm_result.get("news_summary", "")).strip()
    base_advice["sector_impact"] = str(llm_result.get("sector_impact", "")).strip()
    base_advice["policy_impact"] = str(llm_result.get("policy_impact", "")).strip()
    base_advice["news_impact"] = str(llm_result.get("news_impact", "")).strip()
    predictions = ParsePredictionsFromLlm(llm_result, horizon, analysis)
    base_advice["predictions"] = predictions
    base_advice["direction_prediction"] = (
        predictions["near_term"]["prediction"]
        or str(llm_result.get("direction_prediction", "")).strip()
    )
    base_advice["news_conclusion"] = str(llm_result.get("news_conclusion", "")).strip()
    base_advice["pricing_rationale"] = str(llm_result.get("pricing_rationale", "")).strip()

    base_advice["llm_used"] = True
    base_advice["pricing_source"] = pricing_source

    logger.info(
        "DeepSeek 分析完成 — 买入:%s 卖出:%s 加一:%.2f~%.2f 加二:%.2f~%.2f"
        " 卖一:%.2f~%.2f 卖二:%.2f~%.2f",
        buy_action,
        sell_action,
        trade_zones["add_tier1"]["low"],
        trade_zones["add_tier1"]["high"],
        trade_zones["add_tier2"]["low"],
        trade_zones["add_tier2"]["high"],
        trade_zones["sell_tier1"]["low"],
        trade_zones["sell_tier1"]["high"],
        trade_zones["sell_tier2"]["low"],
        trade_zones["sell_tier2"]["high"],
    )
    return base_advice
