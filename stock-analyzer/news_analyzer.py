"""资讯影响分析 — 批量 AI 评分过滤（含 directness 题材错配）。"""

from __future__ import annotations

import json
import re
from typing import Any

import requests

import config
from llm_advisor import IsLlmEnabled
from news_context import (
    BuildCompanyNewsContext,
    ComputeSectorThemeOverlap,
    ComputeStockVsSectorDivergence,
)
from oil_short_advisor import AggregateOilNewsAlert
from oil_short_playbook import IsNonferrousGroup, IsOilShortGroup
from utils import SafeFloat, SetupLogger

logger = SetupLogger(config.LOG_LEVEL)

_LLM_TIMEOUT = 90
_CATALYST_SUBSET_LIMIT = 8
_VALID_IMPACTS = frozenset({"涨", "跌", "中性", "无关"})
_VALID_DIRECTNESS = frozenset({"direct", "indirect", "mismatch"})
_CATEGORY_GUARDS = frozenset({"sector", "policy", "macro", "web_search"})
_IRRELEVANT_KEYWORDS = (
    "国事访问", "欢迎.*访华", "会谈", "会见", "仪式", "致电", "贺电",
    "NBA", "英超", "世界杯",
)
_OIL_DOMAIN_HINT = (
    "原油、布伦特、WTI、OPEC、中东、伊朗、霍尔木兹海峡、地缘冲突、"
    "油服、页岩油、油气开采、海上油气、炼化仅作背景"
)


_NONFERROUS_DOMAIN_HINT = (
    "电解铝、氧化铝、铝价、有色、煤炭、动力煤、限电、产能、水电铝、沪铝"
)


def _DomainLabel() -> str:
    if IsOilShortGroup():
        return "石油/油气/A股"
    if IsNonferrousGroup():
        return "有色/电解铝/A股"
    return "医药/A股"


def _BuildNewsScoringPrompt(stock_name: str, stock_code: str) -> str:
    domain = _DomainLabel()
    domain_extra = ""
    if IsOilShortGroup():
        domain_extra = (
            f"相关域优先：{_OIL_DOMAIN_HINT}。"
            "地缘冲突升级/制裁/通航中断通常偏涨；和谈/缓和/原油大跌通常偏跌。"
            "弱化医药生物类过滤；炼化利润挤压且无上游联动时勿过度标涨。"
        )
    elif IsNonferrousGroup():
        domain_extra = (
            f"相关域优先：{_NONFERROUS_DOMAIN_HINT}。"
            "铝价大涨/限电减产/有色板块普涨通常偏涨；铝价大跌/产能过剩通常偏跌。"
            "弱化医药授权类过滤；煤炭价格仅当明确传导至电解铝成本时再标影响。"
        )
    return (
        f"你是 A 股资讯分析师。请评估每条资讯对 {stock_name}({stock_code}) 的影响。"
        "输出严格 JSON，不要 markdown 代码块。"
        "JSON 格式：{\"items\":[{\"index\":0,\"relevant\":true,\"impact\":\"中性\","
        "\"directness\":\"indirect\",\"impact_reason\":\"30字以内\",\"strength\":1到5}]}"
        "impact 枚举：涨/跌/中性/无关。"
        "directness 枚举：direct（直接指向本公司公告/订单/业绩）/ "
        "indirect（板块或政策环境，对本公司传导弱）/ "
        "mismatch（板块主线与本公司主业或题材不符）。"
        "分类判定规则："
        f"1. stock：仅当直接涉及 {stock_name}/{stock_code} 公告、订单、业绩、监管处罚等"
        "才可 impact=涨/跌 且 directness=direct；"
        "2. sector/policy/macro/web_search 默认 impact=中性、directness=indirect，"
        "除非正文明确提及本公司名称或可验证订单/业绩；"
        "3. 若板块领涨方向与公司主业/配置题材不匹配，标 directness=mismatch，"
        "impact 不得为涨（可为中性或跌）；"
        "4. 若个股涨跌与板块方向明显背离（上下文已给偏离度），"
        "板块类资讯优先 mismatch 而非涨；"
        "5. 禁止无证据写「产业链受益」「题材匹配」；"
        f"6. relevant=false 或 impact=无关 仅用于与{domain}完全无关的内容；"
        f"{domain_extra}"
        "7. strength 表示影响强度，5 最强。仅输出 JSON。"
    )


def _BuildNewsListContext(
    items: list[dict[str, str]],
    company_context: list[str],
) -> str:
    """构建待评分资讯列表文本。"""
    lines = list(company_context)
    lines.extend(["", "【待评估资讯】"])
    for i, item in enumerate(items):
        cat = item.get("category", "stock")
        title = item.get("title", "")
        content = item.get("content", "")[:200]
        lines.append(f"{i}. [{cat}] {title}")
        if content:
            lines.append(f"   摘要：{content}")
    return "\n".join(lines)


def _ParseScoringResult(content: str) -> list[dict[str, Any]] | None:
    """解析 AI 评分 JSON。"""
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        raw_items = data.get("items", data if isinstance(data, list) else [])
        if not isinstance(raw_items, list):
            return None
        return raw_items
    except json.JSONDecodeError as exc:
        logger.error("资讯评分 JSON 解析失败: %s", exc)
        return None


def _CallNewsScoring(context: str, *, retry: int = 1) -> list[dict[str, Any]] | None:
    """调用 DeepSeek 批量评分资讯（失败自动重试）。"""
    url = f"{config.DEEPSEEK_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": _BuildNewsScoringPrompt(config.STOCK_NAME, config.STOCK_CODE)},
            {"role": "user", "content": context},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
        "max_tokens": 2048,
    }
    last_exc: Exception | None = None
    for attempt in range(retry + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=_LLM_TIMEOUT)
            resp.raise_for_status()
            body = resp.json()
            content = body["choices"][0]["message"]["content"]
            parsed = _ParseScoringResult(content)
            if parsed is not None:
                return parsed
            last_exc = ValueError("资讯评分 JSON 解析为空")
        except Exception as exc:
            last_exc = exc
            if attempt < retry:
                logger.warning("资讯 AI 评分第 %d 次失败，重试: %s", attempt + 1, exc)
    logger.error("资讯影响分析 API 失败: %s", last_exc)
    return None


def _RuleFallbackImpactReason(item: dict[str, Any]) -> str:
    """规则兜底 impact_reason：用标题/摘要，不写占位语。"""
    title = str(item.get("title", "")).strip()
    content = str(item.get("content", "")).strip()
    cat = str(item.get("category", "stock"))
    if item.get("is_catalyst"):
        short = title[:40] + ("…" if len(title) > 40 else "")
        return f"板块催化：{short}" if short else "板块催化资讯"
    if cat == "stock" and title:
        return title[:50] + ("…" if len(title) > 50 else "")
    if title:
        return title[:40] + ("…" if len(title) > 40 else "")
    if content:
        return content[:40] + ("…" if len(content) > 40 else "")
    return f"对{_DomainLabel()}环境有间接影响"


def _PickCatalystPriorityItems(
    items: list[dict[str, str]],
    limit: int = _CATALYST_SUBSET_LIMIT,
) -> tuple[list[dict[str, str]], list[int]]:
    """挑选催化优先子集及在原列表中的索引。"""
    indexed = list(enumerate(items))
    catalyst = [(i, it) for i, it in indexed if it.get("is_catalyst")]
    stock_items = [(i, it) for i, it in indexed if it.get("category") == "stock"]
    sector_items = [(i, it) for i, it in indexed if it.get("category") == "sector"]
    picked: list[tuple[int, dict[str, str]]] = []
    seen: set[int] = set()
    for group in (catalyst, stock_items, sector_items):
        for idx, it in group:
            if idx in seen:
                continue
            picked.append((idx, it))
            seen.add(idx)
            if len(picked) >= limit:
                break
        if len(picked) >= limit:
            break
    if not picked:
        picked = indexed[:limit]
    subset = [it for _, it in picked]
    orig_indices = [i for i, _ in picked]
    return subset, orig_indices


def _MergeSubsetScores(
    items: list[dict[str, str]],
    rule_scored: list[dict[str, Any]],
    subset_indices: list[int],
    subset_scores: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """将催化子集 AI 评分合并回完整列表（按原 index）。"""
    result = list(rule_scored)
    score_by_subset_idx: dict[int, dict[str, Any]] = {}
    for s in subset_scores:
        try:
            sub_idx = int(s.get("index", -1))
            if sub_idx >= 0:
                score_by_subset_idx[sub_idx] = s
        except (TypeError, ValueError):
            continue
    for sub_i, orig_i in enumerate(subset_indices):
        score = score_by_subset_idx.get(sub_i)
        if score is None or orig_i >= len(result):
            continue
        cat = str(items[orig_i].get("category", "stock"))
        impact = str(score.get("impact", "中性")).strip()
        if impact not in _VALID_IMPACTS:
            impact = "中性"
        relevant = bool(score.get("relevant", impact != "无关"))
        if impact == "无关":
            relevant = False
        result[orig_i] = {
            **result[orig_i],
            "relevant": relevant,
            "impact": impact,
            "directness": _NormalizeDirectness(score.get("directness"), cat),
            "impact_reason": str(score.get("impact_reason", "")).strip()
            or result[orig_i].get("impact_reason", ""),
            "strength": int(score.get("strength", 1) or 1),
        }
    return result


def _ScoreWithLlm(
    items: list[dict[str, str]],
    company_context: list[str],
    data: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], str]:
    """LLM 评分：全量 → 重试 → 催化子集；返回 scored 与 analysis_source。"""
    context = _BuildNewsListContext(items, company_context)
    scores = _CallNewsScoring(context, retry=1)
    if scores is not None:
        scored = _MergeScores(items, scores)
        scored = _ApplyCategoryRelevanceGuard(scored)
        return _ApplyDirectnessGuard(scored, data), "ai"

    logger.warning("资讯 AI 全量评分失败，尝试催化优先子集")
    subset, orig_indices = _PickCatalystPriorityItems(items)
    if subset:
        sub_context = _BuildNewsListContext(subset, company_context)
        sub_scores = _CallNewsScoring(sub_context, retry=1)
        if sub_scores is not None:
            rule_base = _ApplyCategoryRelevanceGuard(_ApplyRuleFallback(items, data))
            merged = _MergeSubsetScores(items, rule_base, orig_indices, sub_scores)
            return _ApplyDirectnessGuard(merged, data), "ai_partial"

    logger.warning("资讯 AI 评分失败，使用规则兜底")
    scored = _ApplyDirectnessGuard(
        _ApplyCategoryRelevanceGuard(_ApplyRuleFallback(items, data)),
        data,
    )
    return scored, "rule"


def _IsClearlyIrrelevant(title: str, content: str) -> bool:
    """判断资讯是否与目标板块/A股明显无关。"""
    text = f"{title} {content}"
    for pattern in _IRRELEVANT_KEYWORDS:
        if re.search(pattern, text):
            return True
    return False


def _NormalizeDirectness(raw: Any, category: str) -> str:
    """规范化 directness 字段。"""
    value = str(raw or "").strip().lower()
    if value in _VALID_DIRECTNESS:
        return value
    if category == "stock":
        return "direct"
    return "indirect"


def _ApplyCategoryRelevanceGuard(scored: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """对板块/政策/宏观类资讯应用分类保底，避免被 AI 过度标为无关。"""
    domain = _DomainLabel()
    guarded: list[dict[str, Any]] = []
    for item in scored:
        cat = str(item.get("category", "stock"))
        title = str(item.get("title", ""))
        content = str(item.get("content", ""))
        impact = str(item.get("impact", "中性"))
        relevant = bool(item.get("relevant", True))

        if cat in _CATEGORY_GUARDS:
            if _IsClearlyIrrelevant(title, content):
                item = {
                    **item,
                    "relevant": False,
                    "impact": "无关",
                    "directness": "indirect",
                }
            elif not relevant or impact == "无关":
                item = {
                    **item,
                    "relevant": True,
                    "impact": "中性" if impact == "无关" else impact,
                    "directness": _NormalizeDirectness(item.get("directness"), cat),
                    "strength": max(int(item.get("strength", 1) or 1), 1),
                }
                if not item.get("impact_reason"):
                    item["impact_reason"] = f"对{domain}环境有间接影响"
        guarded.append(item)
    return guarded


def _ApplyDirectnessGuard(
    scored: list[dict[str, Any]],
    data: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """后处理 directness，防止板块/政策资讯误标为个股利好。"""
    overlap = ComputeSectorThemeOverlap(data)
    divergence = ComputeStockVsSectorDivergence(data)
    overlap_low = overlap.get("overlap_level") == "低"
    stock_weak = divergence < -1.0

    guarded: list[dict[str, Any]] = []
    for item in scored:
        cat = str(item.get("category", "stock"))
        impact = str(item.get("impact", "中性"))
        directness = _NormalizeDirectness(item.get("directness"), cat)
        strength = int(item.get("strength", 1) or 1)
        reason = str(item.get("impact_reason", "")).strip()

        if cat != "stock" and directness not in _VALID_DIRECTNESS:
            directness = "indirect"

        title_text = f"{item.get('title', '')} {item.get('content', '')}"
        stock_named = config.STOCK_NAME in title_text or config.STOCK_CODE in title_text
        if cat != "stock" and not stock_named:
            if directness == "direct":
                directness = "indirect"
            if overlap_low and cat in ("sector", "policy", "web_search"):
                if "产业链" in reason or "题材匹配" in reason or impact == "涨":
                    directness = "mismatch"
            if stock_weak and cat in _CATEGORY_GUARDS and impact == "涨":
                directness = "mismatch"

        if directness in ("indirect", "mismatch") and impact == "涨":
            impact = "中性"
            strength = min(strength, 2)
            if directness == "mismatch" and not reason:
                reason = "板块主线与本公司主业/题材不匹配，不构成个股利好"

        if (
            directness == "mismatch"
            and stock_weak
            and impact == "中性"
            and divergence < -2.0
        ):
            impact = "跌"
            if not reason:
                reason = "题材错配且个股弱于板块，资金规避"

        item = {
            **item,
            "impact": impact,
            "directness": directness,
            "strength": strength,
            "impact_reason": reason or item.get("impact_reason", ""),
        }
        guarded.append(item)
    return guarded


def _ApplyRuleFallback(
    items: list[dict[str, str]],
    data: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """LLM 不可用时规则兜底。"""
    overlap = ComputeSectorThemeOverlap(data)
    divergence = ComputeStockVsSectorDivergence(data)
    scored: list[dict[str, Any]] = []
    for i, item in enumerate(items):
        cat = item.get("category", "stock")
        title = str(item.get("title", ""))
        content = str(item.get("content", ""))
        clearly_irrelevant = _IsClearlyIrrelevant(title, content)
        if clearly_irrelevant:
            scored.append({
                **item,
                "index": i,
                "relevant": False,
                "impact": "无关",
                "directness": "indirect",
                "impact_reason": f"与{_DomainLabel()}无关",
                "strength": 1,
            })
            continue

        text = f"{title} {content}"
        oil_boost = IsOilShortGroup() and any(
            kw in text
            for kw in (
                "原油", "布伦特", "WTI", "OPEC", "中东", "伊朗",
                "海峡", "地缘", "油服", "页岩油", "油气",
            )
        )
        nf_boost = IsNonferrousGroup() and any(
            kw in text
            for kw in (
                "电解铝", "氧化铝", "铝价", "有色", "煤炭", "动力煤",
                "限电", "产能", "水电铝", "沪铝",
            )
        )

        if cat == "stock":
            directness = "direct"
            impact = "中性"
            strength = 3
        elif oil_boost:
            directness = "indirect"
            if any(k in text for k in ("大跌", "缓和", "通航", "停火", "和谈")):
                impact = "跌"
                strength = 4
            elif any(k in text for k in ("冲突", "制裁", "袭击", "封锁", "减产")):
                impact = "涨"
                strength = 4
            else:
                impact = "中性"
                strength = 3
        elif nf_boost:
            directness = "indirect"
            if any(k in text for k in ("铝价大跌", "大跌", "过剩", "复产放量")):
                impact = "跌"
                strength = 4
            elif any(k in text for k in ("铝价大涨", "限电", "减产", "板块大涨", "普涨")):
                impact = "涨"
                strength = 4
            else:
                impact = "中性"
                strength = 3
        elif overlap.get("overlap_level") == "低" and cat in ("sector", "policy", "web_search"):
            directness = "mismatch"
            impact = "跌" if divergence < -2.0 else "中性"
            strength = 2
        else:
            directness = "indirect"
            impact = "中性"
            strength = 2

        scored.append({
            **item,
            "index": i,
            "relevant": True,
            "impact": impact,
            "directness": directness,
            "impact_reason": _RuleFallbackImpactReason(item),
            "strength": strength,
        })
    return scored


def _MergeScores(
    items: list[dict[str, str]],
    scores: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """将评分合并到资讯条目。"""
    score_map: dict[int, dict[str, Any]] = {}
    for s in scores:
        try:
            idx = int(s.get("index", -1))
            if idx >= 0:
                score_map[idx] = s
        except (TypeError, ValueError):
            continue

    merged: list[dict[str, Any]] = []
    for i, item in enumerate(items):
        score = score_map.get(i, {})
        cat = str(item.get("category", "stock"))
        impact = str(score.get("impact", "中性")).strip()
        if impact not in _VALID_IMPACTS:
            impact = "中性"
        relevant = bool(score.get("relevant", impact != "无关"))
        if impact == "无关":
            relevant = False
        directness = _NormalizeDirectness(score.get("directness"), cat)
        merged.append({
            **item,
            "index": i,
            "relevant": relevant,
            "impact": impact,
            "directness": directness,
            "impact_reason": str(score.get("impact_reason", "")).strip(),
            "strength": int(score.get("strength", 1) or 1),
        })
    return merged


def GetRelevantItems(scored_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """获取有相关性的资讯，按 strength 降序。"""
    relevant = [i for i in scored_items if i.get("relevant") and i.get("impact") != "无关"]
    return sorted(relevant, key=lambda x: x.get("strength", 0), reverse=True)


def GetImpactfulItems(scored_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """获取对股价有直接涨/跌影响的资讯（directness=direct）。"""
    return [
        i for i in scored_items
        if i.get("relevant")
        and i.get("impact") in ("涨", "跌")
        and str(i.get("directness", "direct")) == "direct"
    ]


def GetBackgroundNewsItems(scored_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """获取板块背景/题材错配类资讯。"""
    return [
        i for i in scored_items
        if i.get("relevant")
        and str(i.get("directness", "")) in ("indirect", "mismatch")
    ]


def _CountByCategory(items: list[dict[str, Any]]) -> dict[str, int]:
    """按类别统计相关资讯数量。"""
    counts = {"stock": 0, "sector": 0, "policy": 0, "macro": 0, "web_search": 0}
    for item in items:
        cat = str(item.get("category", "stock"))
        if cat in counts:
            counts[cat] += 1
    return counts


_CATEGORY_SCORE_WEIGHT: dict[str, float] = {
    "stock": 1.0,
    "policy": 0.85,
    "web_search": 0.85,
    "sector": 0.7,
    "macro": 0.5,
}

_CATEGORY_LABEL: dict[str, str] = {
    "stock": "个股",
    "sector": "板块",
    "policy": "政策",
    "web_search": "联网",
    "macro": "宏观",
}


def CalcNewsScore(
    relevant_items: list[dict[str, Any]],
) -> tuple[float, list[str]]:
    """将 AI 评分资讯量化为资讯面分数与信号列表。"""
    score = 0.0
    signals: list[str] = []
    scored_items: list[tuple[float, str]] = []

    for item in relevant_items:
        impact = str(item.get("impact", ""))
        strength = int(item.get("strength", 1) or 1)
        cat = str(item.get("category", "stock"))
        directness = str(item.get("directness", "indirect"))
        cat_weight = _CATEGORY_SCORE_WEIGHT.get(cat, 0.7)
        cat_label = _CATEGORY_LABEL.get(cat, cat)
        title = str(item.get("title", "")).strip()
        short_title = title[:28] + ("…" if len(title) > 28 else "")

        if directness == "direct" and impact in ("涨", "跌"):
            delta = strength * 12.0 * cat_weight
            if impact == "涨":
                score += delta
                label = f"{cat_label}利好"
            else:
                score -= delta
                label = f"{cat_label}利空"
            scored_items.append((delta if impact == "涨" else -delta, f"{label}：{short_title}（强度{strength}）"))
        elif directness == "indirect" and impact in ("涨", "跌", "中性"):
            signals.append(f"板块背景：{short_title}（不直接利好个股）")
        elif directness == "mismatch":
            if impact == "跌":
                delta = strength * 10.0 * cat_weight
                score -= delta
                scored_items.append((-delta, f"题材错配：{short_title}（强度{strength}）"))
            else:
                signals.append(f"题材错配：{short_title}")

    scored_items.sort(key=lambda x: abs(x[0]), reverse=True)
    merged_signals = [s for _, s in scored_items[:6]] + [
        s for s in signals if s not in [x[1] for x in scored_items]
    ]
    seen_keys: set[str] = set()
    deduped_signals: list[str] = []
    for signal in merged_signals:
        body = signal.split("：", 1)[-1].strip() if "：" in signal else signal.strip()
        key = body[:32]
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped_signals.append(signal)
        if len(deduped_signals) >= 6:
            break
    signals = deduped_signals

    score = max(-100.0, min(100.0, score))
    return score, signals[:6]


def AnalyzeNewsImpact(
    news_bundle: dict[str, Any],
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """分析资讯对目标股的影响，写回 news_bundle。"""
    items: list[dict[str, str]] = list(news_bundle.get("items", []))
    if not items:
        news_bundle["scored_items"] = []
        news_bundle["relevant_items"] = []
        news_bundle["impactful_items"] = []
        news_bundle["background_items"] = []
        news_bundle["news_score"] = 0.0
        news_bundle["news_signals"] = []
        news_bundle["analysis_source"] = "rule"
        if IsOilShortGroup():
            news_bundle["oil_news_alert"] = AggregateOilNewsAlert(news_bundle, 0.0)
        return news_bundle

    company_context = BuildCompanyNewsContext(data, news_bundle)
    logger.info("正在分析 %d 条资讯对 %s 的影响...", len(items), config.STOCK_NAME)

    analysis_source = "rule"
    if IsLlmEnabled():
        scored, analysis_source = _ScoreWithLlm(items, company_context, data)
        if analysis_source == "rule":
            logger.warning(
                "LLM 已启用但资讯评分走规则兜底 — 请检查 DEEPSEEK_API_KEY/网络",
            )
    else:
        scored = _ApplyDirectnessGuard(
            _ApplyCategoryRelevanceGuard(_ApplyRuleFallback(items, data)),
            data,
        )

    relevant = GetRelevantItems(scored)
    impactful = GetImpactfulItems(scored)
    background = GetBackgroundNewsItems(scored)
    irrelevant_count = len(items) - len(relevant)
    by_category = _CountByCategory(relevant)
    news_score, news_signals = CalcNewsScore(relevant)
    bullish_count = sum(
        1 for i in relevant
        if i.get("impact") == "涨" and i.get("directness") == "direct"
    )
    bearish_count = sum(
        1 for i in relevant
        if i.get("impact") == "跌" and i.get("directness") == "direct"
    )
    mismatch_count = sum(1 for i in relevant if i.get("directness") == "mismatch")

    overlap = ComputeSectorThemeOverlap(data)
    news_bundle["scored_items"] = scored
    news_bundle["relevant_items"] = relevant
    news_bundle["impactful_items"] = impactful
    news_bundle["background_items"] = background
    news_bundle["news_score"] = news_score
    news_bundle["news_signals"] = news_signals
    news_bundle["analysis_source"] = analysis_source
    news_bundle["sector_theme_overlap"] = overlap
    news_bundle["impact_stats"] = {
        "total": len(items),
        "relevant": len(relevant),
        "impactful": len(impactful),
        "background": len(background),
        "irrelevant": irrelevant_count,
        "by_category": by_category,
        "bullish": bullish_count,
        "bearish": bearish_count,
        "mismatch": mismatch_count,
    }
    if IsOilShortGroup():
        news_bundle["oil_news_alert"] = AggregateOilNewsAlert(news_bundle, news_score)

    logger.info(
        "资讯影响分析 — 有关:%d 直接涨/跌:%d 背景/错配:%d 无关:%d ｜ "
        "题材重叠:%s ｜ 资讯面:%+.1f ｜ 来源:%s",
        len(relevant),
        len(impactful),
        len(background),
        irrelevant_count,
        overlap.get("overlap_level", "?"),
        news_score,
        analysis_source,
    )
    return news_bundle
