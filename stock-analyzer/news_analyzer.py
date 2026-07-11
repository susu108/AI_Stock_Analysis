"""资讯影响分析 — 批量 AI 评分过滤。"""

from __future__ import annotations

import json
import re
from typing import Any

import requests

import config
from llm_advisor import IsLlmEnabled
from utils import SetupLogger

logger = SetupLogger(config.LOG_LEVEL)

_LLM_TIMEOUT = 90
_VALID_IMPACTS = frozenset({"涨", "跌", "中性", "无关"})
_CATEGORY_GUARDS = frozenset({"sector", "policy", "macro", "web_search"})
_IRRELEVANT_KEYWORDS = (
    "国事访问", "欢迎.*访华", "会谈", "会见", "仪式", "致电", "贺电",
    "NBA", "英超", "世界杯",
)


def _BuildNewsScoringPrompt(stock_name: str, stock_code: str) -> str:
    return (
        f"你是 A 股资讯分析师。请评估每条资讯对 {stock_name}({stock_code}) 及其所属"
        f"医药/医疗板块的影响。输出严格 JSON，不要 markdown 代码块。"
        "JSON 格式：{\"items\":[{\"index\":0,\"relevant\":true,\"impact\":\"涨\","
        "\"impact_reason\":\"30字以内影响说明\",\"strength\":1到5}]}"
        "impact 枚举：涨/跌/中性/无关。"
        "分类相关判定规则："
        f"1. stock（个股）：直接涉及 {stock_name}/{stock_code}；"
        "2. sector（板块）：涉及医药/医疗/CRO/仿制药/医疗器械等板块，对个股有间接影响即 relevant=true；"
        "3. policy（政策）：涉及药监/医保/国务院/证监会/A股监管政策，即 relevant=true；"
        "4. macro（宏观）：中国宏观数据（CPI/PMI/利率等）影响市场流动性，即 relevant=true；"
        "5. web_search（联网）：政策/题材/涨跌原因相关搜索结果，与 policy 同级判定。"
        "重要：板块/政策/宏观不可仅因未直接提及股票代码就标为无关，"
        "应评估对医药板块及个股的间接影响，可标 impact=中性。"
        "relevant=false 或 impact=无关 仅用于与医药/A股完全无关的内容。"
        "strength 表示影响强度，5 最强。仅输出 JSON。"
    )


def _BuildNewsListContext(
    items: list[dict[str, str]],
    sector_snapshot: list[str],
) -> str:
    """构建待评分资讯列表文本。"""
    lines = [f"目标股票：{config.STOCK_NAME}({config.STOCK_CODE})", ""]
    if sector_snapshot:
        lines.append("【板块快照】")
        lines.extend(sector_snapshot)
        lines.append("")
    lines.append("【待评估资讯】")
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
                }
            elif not relevant or impact == "无关":
                item = {
                    **item,
                    "relevant": True,
                    "impact": "中性" if impact == "无关" else impact,
                    "strength": max(int(item.get("strength", 1) or 1), 1),
                }
                if not item.get("impact_reason"):
                    item["impact_reason"] = "对医药板块或A股环境有间接影响"
        guarded.append(item)
    return guarded


def _ApplyRuleFallback(items: list[dict[str, str]]) -> list[dict[str, Any]]:
    """LLM 不可用时规则兜底：四类资讯默认相关。"""
    scored: list[dict[str, Any]] = []
    for i, item in enumerate(items):
        cat = item.get("category", "stock")
        if cat == "stock":
            strength = 3
        elif cat == "sector":
            strength = 2
        elif cat == "policy":
            strength = 2
        elif cat == "web_search":
            strength = 3
        else:
            strength = 1
        clearly_irrelevant = _IsClearlyIrrelevant(
            str(item.get("title", "")),
            str(item.get("content", "")),
        )
        scored.append({
            **item,
            "index": i,
            "relevant": not clearly_irrelevant,
            "impact": "无关" if clearly_irrelevant else "中性",
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
        impact = str(score.get("impact", "中性")).strip()
        if impact not in _VALID_IMPACTS:
            impact = "中性"
        relevant = bool(score.get("relevant", impact != "无关"))
        if impact == "无关":
            relevant = False
        merged.append({
            **item,
            "index": i,
            "relevant": relevant,
            "impact": impact,
            "impact_reason": str(score.get("impact_reason", "")).strip(),
            "strength": int(score.get("strength", 1) or 1),
        })
    return merged


def GetRelevantItems(scored_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """获取有相关性的资讯，按 strength 降序。"""
    relevant = [i for i in scored_items if i.get("relevant") and i.get("impact") != "无关"]
    return sorted(relevant, key=lambda x: x.get("strength", 0), reverse=True)


def GetImpactfulItems(scored_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """获取有涨/跌影响的资讯。"""
    return [
        i for i in scored_items
        if i.get("relevant") and i.get("impact") in ("涨", "跌")
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
    impactful = [
        i for i in relevant_items
        if str(i.get("impact", "")) in ("涨", "跌")
    ]
    sorted_items = sorted(
        impactful,
        key=lambda x: int(x.get("strength", 0) or 0),
        reverse=True,
    )

    for item in sorted_items:
        impact = str(item.get("impact", ""))
        strength = int(item.get("strength", 1) or 1)
        cat = str(item.get("category", "stock"))
        cat_weight = _CATEGORY_SCORE_WEIGHT.get(cat, 0.7)
        cat_label = _CATEGORY_LABEL.get(cat, cat)
        delta = strength * 12.0 * cat_weight
        if impact == "涨":
            score += delta
            direction_label = "利好"
        elif impact == "跌":
            score -= delta
            direction_label = "利空"
        else:
            continue

        title = str(item.get("title", "")).strip()
        short_title = title[:28] + ("…" if len(title) > 28 else "")
        signals.append(f"{cat_label}{direction_label}：{short_title}（强度{strength}）")

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
        news_bundle["news_score"] = 0.0
        news_bundle["news_signals"] = []
        return news_bundle

    sector_snapshot = news_bundle.get("sector_snapshot", [])
    logger.info("正在分析 %d 条资讯对 %s 的影响...", len(items), config.STOCK_NAME)

    if IsLlmEnabled():
        context = _BuildNewsListContext(items, sector_snapshot)
        scores = _CallNewsScoring(context)
        if scores is not None:
            scored = _ApplyCategoryRelevanceGuard(_MergeScores(items, scores))
        else:
            logger.warning("资讯 AI 评分失败，使用规则兜底")
            scored = _ApplyCategoryRelevanceGuard(_ApplyRuleFallback(items))
    else:
        scored = _ApplyCategoryRelevanceGuard(_ApplyRuleFallback(items))

    relevant = GetRelevantItems(scored)
    impactful = GetImpactfulItems(scored)
    irrelevant_count = len(items) - len(relevant)
    by_category = _CountByCategory(relevant)
    news_score, news_signals = CalcNewsScore(relevant)
    bullish_count = sum(1 for i in relevant if i.get("impact") == "涨")
    bearish_count = sum(1 for i in relevant if i.get("impact") == "跌")

    news_bundle["scored_items"] = scored
    news_bundle["relevant_items"] = relevant
    news_bundle["impactful_items"] = impactful
    news_bundle["news_score"] = news_score
    news_bundle["news_signals"] = news_signals
    news_bundle["impact_stats"] = {
        "total": len(items),
        "relevant": len(relevant),
        "impactful": len(impactful),
        "irrelevant": irrelevant_count,
        "by_category": by_category,
        "bullish": bullish_count,
        "bearish": bearish_count,
    }

    logger.info(
        "资讯影响分析 — 有关:%d 涨/跌:%d 无关:%d ｜ 个股:%d 板块:%d 政策:%d 宏观:%d 联网:%d ｜ 资讯面:%+.1f",
        len(relevant),
        len(impactful),
        irrelevant_count,
        by_category.get("stock", 0),
        by_category.get("sector", 0),
        by_category.get("policy", 0),
        by_category.get("macro", 0),
        by_category.get("web_search", 0),
        news_score,
    )
    return news_bundle
