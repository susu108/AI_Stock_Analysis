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
from utils import SafeFloat, SetupLogger

logger = SetupLogger(config.LOG_LEVEL)

_LLM_TIMEOUT = 90
_VALID_IMPACTS = frozenset({"涨", "跌", "中性", "无关"})
_VALID_DIRECTNESS = frozenset({"direct", "indirect", "mismatch"})
_CATEGORY_GUARDS = frozenset({"sector", "policy", "macro", "web_search"})
_IRRELEVANT_KEYWORDS = (
    "国事访问", "欢迎.*访华", "会谈", "会见", "仪式", "致电", "贺电",
    "NBA", "英超", "世界杯",
)


def _BuildNewsScoringPrompt(stock_name: str, stock_code: str) -> str:
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
        "6. relevant=false 或 impact=无关 仅用于与医药/A股完全无关的内容；"
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


def _CallNewsScoring(context: str) -> list[dict[str, Any]] | None:
    """调用 DeepSeek 批量评分资讯。"""
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
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=_LLM_TIMEOUT)
        resp.raise_for_status()
        body = resp.json()
        content = body["choices"][0]["message"]["content"]
        return _ParseScoringResult(content)
    except Exception as exc:
        logger.error("资讯影响分析 API 失败: %s", exc)
        return None


def _IsClearlyIrrelevant(title: str, content: str) -> bool:
    """判断资讯是否与医药/A股明显无关。"""
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
                    item["impact_reason"] = "对医药板块或A股环境有间接影响"
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
                "impact_reason": "与医药/A股无关",
                "strength": 1,
            })
            continue

        if cat == "stock":
            directness = "direct"
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
            "impact_reason": "规则兜底，建议启用 AI 获取精准影响",
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
    signals = [s for _, s in scored_items[:6]] + [
        s for s in signals if s not in [x[1] for x in scored_items]
    ][:6]

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
        return news_bundle

    company_context = BuildCompanyNewsContext(data, news_bundle)
    logger.info("正在分析 %d 条资讯对 %s 的影响...", len(items), config.STOCK_NAME)

    if IsLlmEnabled():
        context = _BuildNewsListContext(items, company_context)
        scores = _CallNewsScoring(context)
        if scores is not None:
            scored = _MergeScores(items, scores)
            scored = _ApplyCategoryRelevanceGuard(scored)
            scored = _ApplyDirectnessGuard(scored, data)
        else:
            logger.warning("资讯 AI 评分失败，使用规则兜底")
            scored = _ApplyDirectnessGuard(
                _ApplyCategoryRelevanceGuard(_ApplyRuleFallback(items, data)),
                data,
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

    logger.info(
        "资讯影响分析 — 有关:%d 直接涨/跌:%d 背景/错配:%d 无关:%d ｜ "
        "题材重叠:%s ｜ 资讯面:%+.1f",
        len(relevant),
        len(impactful),
        len(background),
        irrelevant_count,
        overlap.get("overlap_level", "?"),
        news_score,
    )
    return news_bundle
