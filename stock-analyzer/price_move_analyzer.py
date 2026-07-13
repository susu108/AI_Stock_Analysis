"""涨跌归因分析 — 五维因果拆解（独立 LLM 调用）。"""

from __future__ import annotations

import json
import re
from typing import Any

import requests

import config
from llm_advisor import IsLlmEnabled
from utils import SetupLogger

logger = SetupLogger(config.LOG_LEVEL)

_LLM_TIMEOUT = 120
_VALID_DRIVERS = frozenset({"policy", "sector", "technical", "sentiment", "mixed"})
_DIMENSION_IDS = ("catalyst", "sector", "technical", "sentiment", "risks")
_DEFAULT_DISCLAIMER = (
    "风险提示：仅行情逻辑复盘，不构成任何投资建议。"
    "归因基于已采集资讯与行情数据，可能存在信息滞后或证据不足。"
)


def ShouldSkipFullAnalysis(move_context: dict[str, Any]) -> bool:
    """涨跌幅过小且无量能异常时跳过完整 LLM 归因。"""
    change_pct = abs(float(move_context.get("change_pct", 0) or 0))
    volume_ratio = float(move_context.get("volume_ratio", 1.0) or 1.0)
    threshold = config.PRICE_MOVE_MIN_PCT
    if change_pct >= threshold:
        return False
    if volume_ratio >= 1.5:
        return False
    return True


def BuildMoveContextPrompt(move_context: dict[str, Any]) -> str:
    """将结构化上下文序列化为 LLM 输入。"""
    overlap = move_context.get("sector_theme_overlap") or {}
    lines = [
        f"目标：{move_context.get('stock_name')}({move_context.get('stock_code')})",
        f"日期：{move_context.get('move_date')}",
        f"涨跌：{move_context.get('move_direction')} {move_context.get('change_pct'):+.2f}%"
        f"（{move_context.get('change_amt'):+.2f}元）",
        f"现价：{move_context.get('price')} 昨收：{move_context.get('prev_close')}",
        f"成交额：{move_context.get('amount_text')} 量比：{move_context.get('volume_ratio'):.2f}",
        f"流通市值：{move_context.get('circ_mv_yi')}亿（{move_context.get('mv_tier')}）",
        f"昨日涨跌：{move_context.get('prev_day_change_pct'):+.2f}%",
        f"箱体叙事：{move_context.get('box_break_narrative')}",
        f"主力净流入：{move_context.get('main_net_inflow'):.0f}元",
        f"融资余额变动：{move_context.get('margin_balance_change'):.0f}元",
        f"个股题材：{', '.join(move_context.get('stock_themes') or []) or '未配置'}",
    ]
    if move_context.get("stock_business"):
        lines.append(f"公司主业：{move_context.get('stock_business')}")
    lines.append(
        f"概念题材重叠度：{overlap.get('overlap_level', '未知')}；"
        f"个股相对概念均值偏离：{move_context.get('stock_vs_sector_divergence', 0):+.2f}%"
    )
    lines.extend(["", "【板块领涨】"])
    for leader in move_context.get("sector_leaders") or []:
        lines.append(
            f"- [{leader.get('type')}] {leader.get('name')} "
            f"{leader.get('change_pct'):+.2f}%"
        )
    if move_context.get("sector_snapshot"):
        lines.extend(["", "【板块快照】"])
        lines.extend(move_context.get("sector_snapshot") or [])

    lines.extend(["", "【技术信号】"])
    for sig in (move_context.get("tech_signals") or [])[:6]:
        lines.append(f"- {sig}")
    lines.extend(["", "【资金信号】"])
    for sig in (move_context.get("fund_signals") or [])[:4]:
        lines.append(f"- {sig}")

    lines.extend(["", "【个股直接催化资讯 — directness=direct】"])
    for item in move_context.get("impactful_news") or []:
        lines.append(
            f"- [{item.get('category')}|{item.get('impact')}|direct] {item.get('title')} "
            f"— {item.get('impact_reason')}"
        )
    if not (move_context.get("impactful_news") or []):
        lines.append("- 无直接催化资讯")

    lines.extend(["", "【板块背景/题材错配 — 不得等同个股利好】"])
    for item in move_context.get("background_news") or []:
        lines.append(
            f"- [{item.get('category')}|{item.get('directness')}|{item.get('impact')}] "
            f"{item.get('title')} — {item.get('impact_reason')}"
        )
    if not (move_context.get("background_news") or []):
        lines.append("- 无板块背景/错配资讯")

    other_news = [
        i for i in (move_context.get("relevant_news") or [])
        if i not in (move_context.get("impactful_news") or [])
        and i not in (move_context.get("background_news") or [])
    ]
    if other_news:
        lines.extend(["", "【其他相关资讯】"])
        for item in other_news:
            lines.append(f"- [{item.get('category')}] {item.get('title')}")

    lines.extend(["", "【联网搜索】"])
    for item in move_context.get("web_search_hits") or []:
        lines.append(f"- {item.get('title')} | {item.get('source')} | {item.get('url')}")
        snippet = item.get("content", "")
        if snippet:
            lines.append(f"  {snippet[:200]}")

    return "\n".join(lines)


def _BuildPriceMoveSystemPrompt() -> str:
    return (
        "你是 A 股涨跌归因分析师。基于用户提供的行情、资讯、板块与资金数据，"
        "复盘当日（或最近交易日）股价涨跌的因果链条。"
        "输出严格 JSON，不要 markdown 代码块。"
        "JSON 结构："
        "{"
        '"move_date":"YYYY-MM-DD",'
        '"move_direction":"涨|跌|平",'
        '"move_pct":数字,'
        '"headline":"30字以内一句话归因",'
        '"disclaimer":"风险提示文案",'
        '"primary_driver":"policy|sector|technical|sentiment|mixed",'
        '"dimensions":[{'
        '"id":"catalyst|sector|technical|sentiment|risks",'
        '"title":"章节标题",'
        '"points":[{"subtitle":"小标题","content":"60~120字分析","evidence":"引用的数据/资讯","source_url":""}],'
        '"limitations":"该维度的限制/边界（可为空字符串）"'
        "}],"
        '"evidence_gaps":["未能验证的点"]'
        "}"
        "要求："
        "1. dimensions 必须包含 catalyst/sector/technical/sentiment/risks 五维；"
        "2. 每维 2~4 个 points，content 须具体引用上下文中的数字、价位、资讯标题；"
        "3. 无证据时不编造，写入 evidence_gaps；"
        "4. catalyst 仅引用【个股直接催化】资讯；无 direct 催化时写「无直接催化，板块利好未传导」；"
        "5. sector 维须对比今日领涨板块与公司主业/题材是否同赛道，说明个股是否跟涨及分化原因；"
        "6. 【板块背景/题材错配】资讯只可写入 sector 维，不得写入 catalyst 作为个股利好；"
        "7. risks 维分析利好兑现、套牢区、杠杆抛压、题材错配等后续隐患；"
        "8. 涨跌幅为负时重点分析下跌原因；"
        "9. 仅输出 JSON。"
    )


def _ParsePriceMoveJson(content: str) -> dict[str, Any] | None:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
        return None
    except json.JSONDecodeError as exc:
        logger.error("涨跌归因 JSON 解析失败: %s", exc)
        return None


def _CallPriceMoveLlm(context: str) -> dict[str, Any] | None:
    url = f"{config.DEEPSEEK_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": _BuildPriceMoveSystemPrompt()},
            {"role": "user", "content": context},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.25,
        "max_tokens": 4096,
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=_LLM_TIMEOUT)
        resp.raise_for_status()
        body = resp.json()
        content = body["choices"][0]["message"]["content"]
        return _ParsePriceMoveJson(content)
    except requests.Timeout:
        logger.error("涨跌归因 LLM 超时（%ds）", _LLM_TIMEOUT)
        return None
    except requests.HTTPError as exc:
        detail = exc.response.text[:200] if exc.response is not None else ""
        logger.error("涨跌归因 LLM HTTP 错误: %s — %s", exc, detail)
        return None
    except (KeyError, IndexError, TypeError) as exc:
        logger.error("涨跌归因 LLM 响应解析失败: %s", exc)
        return None
    except Exception as exc:
        logger.error("涨跌归因 LLM 异常: %s", exc)
        return None


def _NormalizeDimensions(raw_dims: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_dims, list):
        return []
    normalized: list[dict[str, Any]] = []
    for dim in raw_dims:
        if not isinstance(dim, dict):
            continue
        dim_id = str(dim.get("id", "")).strip()
        if dim_id not in _DIMENSION_IDS:
            continue
        points_raw = dim.get("points", [])
        points: list[dict[str, str]] = []
        if isinstance(points_raw, list):
            for pt in points_raw:
                if not isinstance(pt, dict):
                    continue
                content = str(pt.get("content", "")).strip()
                if not content:
                    continue
                points.append({
                    "subtitle": str(pt.get("subtitle", "")).strip(),
                    "content": content,
                    "evidence": str(pt.get("evidence", "")).strip(),
                    "source_url": str(pt.get("source_url", "")).strip(),
                })
        normalized.append({
            "id": dim_id,
            "title": str(dim.get("title", "")).strip() or dim_id,
            "points": points,
            "limitations": str(dim.get("limitations", "")).strip(),
        })
    return normalized


def _EnsureAllDimensions(dims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """补齐缺失维度。"""
    titles = {
        "catalyst": "核心直接催化",
        "sector": "板块资金共振",
        "technical": "技术面超跌修复",
        "sentiment": "市场情绪与筹码",
        "risks": "关键利空/隐患",
    }
    existing = {d["id"]: d for d in dims}
    result: list[dict[str, Any]] = []
    for dim_id in _DIMENSION_IDS:
        if dim_id in existing:
            result.append(existing[dim_id])
        else:
            result.append({
                "id": dim_id,
                "title": titles[dim_id],
                "points": [],
                "limitations": "",
            })
    return result


def BuildRulePriceMoveFallback(move_context: dict[str, Any]) -> dict[str, Any]:
    """LLM 不可用时的规则兜底归因。"""
    direction = str(move_context.get("move_direction", "平"))
    change_pct = float(move_context.get("change_pct", 0) or 0)
    points_catalyst: list[dict[str, str]] = []
    for item in move_context.get("impactful_news") or []:
        points_catalyst.append({
            "subtitle": str(item.get("category", "资讯")),
            "content": f"{item.get('title')} — {item.get('impact_reason') or item.get('content', '')[:80]}",
            "evidence": item.get("title", ""),
            "source_url": item.get("url", ""),
        })
    if not points_catalyst:
        points_catalyst.append({
            "subtitle": "直接催化",
            "content": "无直接催化资讯，板块/政策利好未明确传导至个股",
            "evidence": "资讯 directness 分层",
            "source_url": "",
        })

    sector_points: list[dict[str, str]] = []
    for leader in (move_context.get("sector_leaders") or [])[:3]:
        overlap = move_context.get("sector_theme_overlap") or {}
        overlap_level = overlap.get("overlap_level", "未知")
        divergence = float(move_context.get("stock_vs_sector_divergence", 0) or 0)
        same_track = overlap_level in ("高", "中")
        if same_track and change_pct > 0:
            resonance = "板块共振推升个股"
        elif not same_track:
            resonance = f"板块领涨与本公司主业重叠度{overlap_level}，个股{'独立走强' if divergence > 0 else '未跟涨板块'}"
        else:
            resonance = "板块拖累个股"
        sector_points.append({
            "subtitle": leader.get("type", "板块"),
            "content": f"{leader.get('name')} 涨跌幅 {leader.get('change_pct'):+.2f}%，{resonance}",
            "evidence": "板块行情数据",
            "source_url": "",
        })
    for item in move_context.get("background_news") or [][:2]:
        directness = str(item.get("directness", "indirect"))
        label = "题材错配" if directness == "mismatch" else "板块背景"
        sector_points.append({
            "subtitle": label,
            "content": f"{item.get('title')} — {item.get('impact_reason') or item.get('content', '')[:80]}",
            "evidence": item.get("title", ""),
            "source_url": item.get("url", ""),
        })

    tech_points = [{
        "subtitle": "技术形态",
        "content": str(move_context.get("box_break_narrative", "")),
        "evidence": "K线箱体/高低点",
        "source_url": "",
    }]
    for sig in (move_context.get("tech_signals") or [])[:2]:
        tech_points.append({
            "subtitle": "技术信号",
            "content": sig,
            "evidence": "技术指标",
            "source_url": "",
        })

    sentiment_points: list[dict[str, str]] = [{
        "subtitle": "量能",
        "content": f"成交额 {move_context.get('amount_text')}，量比 {move_context.get('volume_ratio', 1):.2f}",
        "evidence": "实时行情",
        "source_url": "",
    }]
    if move_context.get("mv_tier"):
        sentiment_points.append({
            "subtitle": "市值特征",
            "content": f"流通市值约 {move_context.get('circ_mv_yi')} 亿，属{move_context.get('mv_tier')}",
            "evidence": "流通市值",
            "source_url": "",
        })
    margin_chg = float(move_context.get("margin_balance_change", 0) or 0)
    if margin_chg != 0:
        sentiment_points.append({
            "subtitle": "融资盘",
            "content": f"融资余额较前日变动约 {margin_chg:+.0f} 元",
            "evidence": "融资融券数据",
            "source_url": "",
        })

    risk_points: list[dict[str, str]] = [{
        "subtitle": "题材风险",
        "content": "若上涨由政策/题材驱动，需警惕「买预期、卖事实」的获利了结窗口",
        "evidence": "A股常见题材节奏",
        "source_url": "",
    }]
    mismatch_items = [
        i for i in (move_context.get("background_news") or [])
        if str(i.get("directness", "")) == "mismatch"
    ]
    if mismatch_items:
        risk_points.append({
            "subtitle": "题材错配",
            "content": f"板块主线与本公司主业不符：{mismatch_items[0].get('title', '')}",
            "evidence": "题材重叠度分析",
            "source_url": "",
        })
    if len(move_context.get("impactful_news") or []) == 0:
        risk_points.append({
            "subtitle": "证据不足",
            "content": "未采集到足够直接影响涨跌的个股资讯/政策证据，归因可能不完整",
            "evidence": "资讯采集结果",
            "source_url": "",
        })

    evidence_gaps: list[str] = []
    if len(move_context.get("impactful_news") or []) == 0:
        evidence_gaps.append("缺少直接影响涨跌的个股直接资讯/政策证据")
    if margin_chg == 0:
        evidence_gaps.append("融资融券变动数据不可用")

    primary = "mixed"
    direct_catalyst = [
        i for i in (move_context.get("impactful_news") or [])
        if str(i.get("impact", "")) in ("涨", "跌")
    ]
    if direct_catalyst:
        primary = "policy"
    elif sector_points:
        primary = "sector"

    headline = f"{'上涨' if direction == '涨' else '下跌' if direction == '跌' else '震荡'}{abs(change_pct):.2f}%"
    if direct_catalyst:
        headline += "，个股直接资讯驱动为主"
    elif sector_points:
        headline += "，板块环境/分化为主"
    else:
        headline += "，技术/资金因素为主"

    return {
        "move_date": move_context.get("move_date", ""),
        "move_direction": direction,
        "move_pct": change_pct,
        "headline": headline,
        "disclaimer": _DEFAULT_DISCLAIMER,
        "primary_driver": primary,
        "dimensions": _EnsureAllDimensions([
            {"id": "catalyst", "title": "核心直接催化", "points": points_catalyst, "limitations": ""},
            {"id": "sector", "title": "板块资金共振", "points": sector_points, "limitations": ""},
            {"id": "technical", "title": "技术面超跌修复", "points": tech_points, "limitations": ""},
            {"id": "sentiment", "title": "市场情绪与筹码", "points": sentiment_points, "limitations": ""},
            {"id": "risks", "title": "关键利空/隐患", "points": risk_points, "limitations": ""},
        ]),
        "evidence_gaps": evidence_gaps,
        "source": "rule_fallback",
        "brief": ShouldSkipFullAnalysis(move_context),
    }


def BuildBriefPriceMove(move_context: dict[str, Any]) -> dict[str, Any]:
    """涨跌幅过小时的简版归因。"""
    result = BuildRulePriceMoveFallback(move_context)
    result["headline"] = (
        f"涨跌幅 {move_context.get('change_pct'):+.2f}% 未达归因阈值"
        f"（{config.PRICE_MOVE_MIN_PCT}%），暂无深度拆解"
    )
    result["brief"] = True
    result["source"] = "brief"
    return result


def AnalyzePriceMove(move_context: dict[str, Any]) -> dict[str, Any]:
    """分析涨跌归因，优先 LLM，失败则规则兜底。"""
    if ShouldSkipFullAnalysis(move_context):
        logger.info("涨跌幅未达阈值，输出简版涨跌归因")
        return BuildBriefPriceMove(move_context)

    if not IsLlmEnabled():
        logger.info("LLM 未启用，使用规则兜底涨跌归因")
        return BuildRulePriceMoveFallback(move_context)

    prompt = BuildMoveContextPrompt(move_context)
    logger.info("正在调用 DeepSeek 生成涨跌归因拆解...")
    llm_result = _CallPriceMoveLlm(prompt)

    if llm_result is None:
        logger.warning("涨跌归因 LLM 失败，回退规则引擎")
        return BuildRulePriceMoveFallback(move_context)

    direction = str(llm_result.get("move_direction", move_context.get("move_direction", "平")))
    if direction not in ("涨", "跌", "平"):
        direction = str(move_context.get("move_direction", "平"))

    primary = str(llm_result.get("primary_driver", "mixed"))
    if primary not in _VALID_DRIVERS:
        primary = "mixed"

    dims = _EnsureAllDimensions(_NormalizeDimensions(llm_result.get("dimensions")))

    gaps_raw = llm_result.get("evidence_gaps", [])
    evidence_gaps: list[str] = []
    if isinstance(gaps_raw, list):
        evidence_gaps = [str(g).strip() for g in gaps_raw if str(g).strip()]

    result: dict[str, Any] = {
        "move_date": str(llm_result.get("move_date", move_context.get("move_date", ""))),
        "move_direction": direction,
        "move_pct": float(llm_result.get("move_pct", move_context.get("change_pct", 0)) or 0),
        "headline": str(llm_result.get("headline", "")).strip() or "涨跌归因分析",
        "disclaimer": str(llm_result.get("disclaimer", "")).strip() or _DEFAULT_DISCLAIMER,
        "primary_driver": primary,
        "dimensions": dims,
        "evidence_gaps": evidence_gaps,
        "source": "llm",
        "brief": False,
    }
    logger.info(
        "涨跌归因完成 — 主因:%s 五维:%d evidence_gaps:%d",
        primary,
        len(dims),
        len(evidence_gaps),
    )
    return result
