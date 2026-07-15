"""催化资讯专项 AI 解读 — 说明对监控股的传导与操作建议。"""

from __future__ import annotations

import json
import re
from typing import Any

import requests

import config
from llm_advisor import IsLlmEnabled
from sector_catalyst_watch import IsCatalystText
from utils import SetupLogger

logger = SetupLogger(config.LOG_LEVEL)

_LLM_TIMEOUT = 60
_VALID_IMPACTS = frozenset({"涨", "跌", "中性"})


def _FindTriggerItem(
    news_bundle: dict[str, Any],
    trigger_headline: str,
) -> dict[str, Any] | None:
    """从 bundle 中定位触发资讯条目。"""
    headline = trigger_headline.strip()
    items = list(news_bundle.get("relevant_items") or news_bundle.get("items") or [])
    if headline:
        for item in items:
            if headline in str(item.get("title", "")):
                return item
    for item in items:
        if item.get("is_catalyst") or IsCatalystText(
            str(item.get("title", "")), str(item.get("content", "")),
        ):
            return item
    for item in items:
        if str(item.get("impact", "")) in ("涨", "跌"):
            return item
    return items[0] if items else None


def _BuildCatalystPrompt(
    item: dict[str, Any],
    data: dict[str, Any] | None,
) -> str:
    """构建催化解读 user prompt。"""
    rt = (data or {}).get("realtime") or {}
    change_pct = rt.get("change_pct", "")
    lines = [
        f"监控标的：{config.STOCK_NAME}({config.STOCK_CODE})",
        f"主业/题材：{config.STOCK_BUSINESS or '未配置'}",
        f"配置题材：{', '.join(config.STOCK_THEMES or [])}",
        f"现价涨跌：{change_pct}%",
        "",
        "【触发资讯】",
        f"标题：{item.get('title', '')}",
        f"摘要：{str(item.get('content', ''))[:300]}",
        f"分类：{item.get('category', '')}",
        f"已有影响判断：{item.get('impact', '中性')} / {item.get('directness', '')}",
        f"已有说明：{item.get('impact_reason', '')}",
    ]
    return "\n".join(lines)


def _ParseCatalystLlmResult(content: str) -> dict[str, str] | None:
    """解析催化解读 JSON。"""
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        summary = str(data.get("catalyst_summary", "")).strip()
        impact = str(data.get("impact_on_stock", "中性")).strip()
        hint = str(data.get("action_hint", "")).strip()
        if impact not in _VALID_IMPACTS:
            impact = "中性"
        if not summary:
            return None
        return {
            "catalyst_summary": summary[:150],
            "impact_on_stock": impact,
            "action_hint": hint[:80],
        }
    except json.JSONDecodeError as exc:
        logger.error("催化解读 JSON 解析失败: %s", exc)
        return None


def _CallCatalystLlm(prompt: str) -> dict[str, str] | None:
    """调用 DeepSeek 生成催化专项解读。"""
    url = f"{config.DEEPSEEK_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    system = (
        f"你是 A 股短线分析师。请解读下方资讯对 {config.STOCK_NAME}({config.STOCK_CODE}) 的影响。"
        "输出严格 JSON，不要 markdown。"
        "字段：catalyst_summary（80~120字，说明资讯内容及对本公司传导路径）；"
        "impact_on_stock（涨/跌/中性）；"
        "action_hint（一句操作建议，如「板块情绪利好，勿当个股直接订单」）。"
        "禁止无证据写「产业链直接受益」；同行重磅默认 indirect 传导。"
    )
    payload = {
        "model": config.DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.3,
        "max_tokens": 512,
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=_LLM_TIMEOUT)
        resp.raise_for_status()
        body = resp.json()
        content = body["choices"][0]["message"]["content"]
        return _ParseCatalystLlmResult(content)
    except Exception as exc:
        logger.error("催化专项 AI 解读失败: %s", exc)
        return None


def EnrichCatalystWithLlm(
    news_bundle: dict[str, Any],
    data: dict[str, Any] | None = None,
    trigger_headline: str = "",
) -> dict[str, Any]:
    """对触发/催化资讯生成专项 AI 解读，写入 catalyst_llm。"""
    if not IsLlmEnabled():
        return news_bundle
    if news_bundle.get("catalyst_llm"):
        existing = news_bundle["catalyst_llm"]
        trig = str(existing.get("trigger_title", "")).strip()
        if not trigger_headline or (
            trig and trigger_headline in trig
        ):
            return news_bundle

    item = _FindTriggerItem(news_bundle, trigger_headline)
    if item is None:
        return news_bundle

    prompt = _BuildCatalystPrompt(item, data)
    result = _CallCatalystLlm(prompt)
    if result:
        news_bundle["catalyst_llm"] = {
            **result,
            "trigger_title": str(item.get("title", ""))[:80],
            "trigger_time": str(
                item.get("time") or item.get("pub_time") or ""
            )[:19],
        }
        logger.info(
            "催化 AI 解读完成 — %s 影响:%s",
            config.STOCK_CODE,
            result.get("impact_on_stock"),
        )
    return news_bundle
