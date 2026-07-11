"""持仓建议模块 — 分账户 AI 分析与专报。"""

from __future__ import annotations

import json
import re
from typing import Any

import requests

import config
from llm_advisor import IsLlmEnabled
from portfolio import CalcLotSummary, CalcPortfolioSummary, LoadPortfolio
from utils import SetupLogger

logger = SetupLogger(config.LOG_LEVEL)

_LLM_TIMEOUT = 90


def _BuildPortfolioPrompt() -> str:
    return (
        "你是 A 股持仓顾问。用户持有同一股票的多笔分账户仓位，请分别给出建议。"
        "输出严格 JSON，不要 markdown 代码块。"
        "JSON 格式："
        "{\"accounts\":[{\"account\":\"账户名\",\"strategy\":\"80字策略\","
        "\"action\":\"减仓/持有/观望/做T\",\"target_price\":数字,"
        "\"note\":\"补充说明\","
        "\"t_trade\":{\"enabled\":布尔,\"buy_price\":数字,\"sell_price\":数字,"
        "\"shares\":整数,\"note\":\"做T说明\"}}],"
        "\"overall_breakeven\":数字,\"recovery_summary\":\"整体回本路径80字\"}"
        "原则：深套小仓优先减亏/反弹减仓，主力仓可逢支撑做T（T+1）。"
        "仅输出 JSON。"
    )


def _BuildPortfolioContext(
    data: dict[str, Any],
    analysis: dict[str, Any],
    current_price: float,
) -> str:
    """构建持仓分析上下文。"""
    summary = CalcPortfolioSummary(current_price)
    if summary is None:
        return ""

    lines = [
        f"股票：{config.STOCK_NAME}({config.STOCK_CODE})",
        f"现价：{current_price:.2f}",
        f"加权成本：{summary.weighted_cost:.2f}",
        f"整体回本需涨：{summary.distance_to_breakeven_pct:.1f}%",
        "",
        "【分账户持仓】",
    ]
    for lot in summary.lots:
        info = CalcLotSummary(current_price, lot)
        lines.append(
            f"- {lot.account}：成本 {lot.cost:.3f} × {lot.shares}股，"
            f"浮盈亏 {info['pnl']:+.2f}元（{info['pnl_pct']:+.1f}%），"
            f"距回本需涨 {info['distance_to_breakeven_pct']:.1f}%"
        )

    lines.extend([
        "",
        "【技术面参考】",
        *[f"- {s}" for s in analysis.get("tech_signals", [])[:4]],
        "",
        "【支撑压力参考】",
        f"- 综合评分：{analysis.get('weighted_score', 0):+.1f}",
        f"- 预测方向：{analysis.get('direction', '')}",
    ])
    rt = data.get("realtime") or {}
    if rt:
        lines.append(f"- 涨跌幅：{rt.get('change_pct', 0)}%")
    return "\n".join(lines)


def _ParsePortfolioJson(content: str) -> dict[str, Any] | None:
    """解析持仓建议 JSON。"""
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        result = json.loads(text)
        return result if isinstance(result, dict) else None
    except json.JSONDecodeError as exc:
        logger.error("持仓建议 JSON 解析失败: %s", exc)
        return None


def _CallPortfolioLlm(context: str) -> dict[str, Any] | None:
    """调用 DeepSeek 生成分账户持仓建议。"""
    url = f"{config.DEEPSEEK_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": _BuildPortfolioPrompt()},
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
        return _ParsePortfolioJson(content)
    except Exception as exc:
        logger.error("持仓建议 API 失败: %s", exc)
        return None


def _BuildRulePortfolioAdvice(current_price: float) -> dict[str, Any]:
    """规则引擎生成分账户持仓建议。"""
    lots = LoadPortfolio()
    accounts: list[dict[str, Any]] = []
    for lot in lots:
        info = CalcLotSummary(current_price, lot)
        pnl_pct = info["pnl_pct"]
        if pnl_pct < -30:
            strategy = "深套小仓，反弹至成本附近分批减仓减亏，不宜加仓"
            action = "减仓"
        elif abs(pnl_pct) < 5:
            strategy = "接近成本主力仓，可逢支撑低吸、逢压力高抛做T"
            action = "做T"
        else:
            strategy = "持有观望，等待趋势明朗"
            action = "持有"
        accounts.append({
            "account": lot.account,
            "cost": lot.cost,
            "shares": lot.shares,
            "pnl_pct": pnl_pct,
            "strategy": strategy,
            "action": action,
            "target_price": lot.cost,
            "note": "",
            "t_trade": {"enabled": action == "做T"},
        })

    summary = CalcPortfolioSummary(current_price)
    return {
        "accounts": accounts,
        "overall_breakeven": summary.breakeven_price if summary else 0.0,
        "recovery_summary": "规则引擎建议，建议启用 AI 获取更精准分账户策略",
        "llm_used": False,
        "current_price": current_price,
    }


def GeneratePortfolioAdvice(
    data: dict[str, Any],
    analysis: dict[str, Any],
) -> dict[str, Any]:
    """生成分账户持仓建议。"""
    rt = data.get("realtime") or {}
    indicators = analysis.get("indicators", {})
    current_price = float(rt.get("price", indicators.get("price", 0)))

    lots = LoadPortfolio()
    if not lots:
        logger.warning("未配置持仓，无法生成持仓建议")
        return {"accounts": [], "llm_used": False, "current_price": current_price}

    summary = CalcPortfolioSummary(current_price)
    base: dict[str, Any] = {
        "accounts": [],
        "overall_breakeven": summary.breakeven_price if summary else 0.0,
        "recovery_summary": "",
        "llm_used": False,
        "current_price": current_price,
        "portfolio": summary,
    }

    for lot in lots:
        info = CalcLotSummary(current_price, lot)
        base["accounts"].append({
            "account": lot.account,
            "cost": lot.cost,
            "shares": lot.shares,
            "pnl_pct": info["pnl_pct"],
            "pnl": info["pnl"],
            "market_value": info["market_value"],
            "distance_to_breakeven_pct": info["distance_to_breakeven_pct"],
        })

    if not IsLlmEnabled():
        rule = _BuildRulePortfolioAdvice(current_price)
        for i, acct in enumerate(base["accounts"]):
            if i < len(rule["accounts"]):
                acct.update({
                    "strategy": rule["accounts"][i].get("strategy", ""),
                    "action": rule["accounts"][i].get("action", "持有"),
                    "target_price": rule["accounts"][i].get("target_price", acct["cost"]),
                    "note": rule["accounts"][i].get("note", ""),
                    "t_trade": rule["accounts"][i].get("t_trade", {"enabled": False}),
                })
        base["recovery_summary"] = rule.get("recovery_summary", "")
        return base

    context = _BuildPortfolioContext(data, analysis, current_price)
    logger.info("正在调用 DeepSeek 生成分账户持仓建议...")
    llm_result = _CallPortfolioLlm(context)

    if llm_result is None:
        logger.warning("持仓 AI 分析失败，使用规则引擎")
        return _BuildRulePortfolioAdvice(current_price)

    llm_accounts = llm_result.get("accounts", [])
    acct_map = {str(a.get("account", "")): a for a in llm_accounts if isinstance(a, dict)}

    for acct in base["accounts"]:
        llm_acct = acct_map.get(acct["account"], {})
        acct["strategy"] = str(llm_acct.get("strategy", "")).strip()
        acct["action"] = str(llm_acct.get("action", "持有")).strip()
        acct["target_price"] = float(llm_acct.get("target_price", acct["cost"]) or acct["cost"])
        acct["note"] = str(llm_acct.get("note", "")).strip()
        t_trade = llm_acct.get("t_trade")
        if isinstance(t_trade, dict):
            acct["t_trade"] = {
                "enabled": bool(t_trade.get("enabled", False)),
                "buy_price": float(t_trade.get("buy_price", 0) or 0),
                "sell_price": float(t_trade.get("sell_price", 0) or 0),
                "shares": int(t_trade.get("shares", 0) or 0),
                "note": str(t_trade.get("note", "")).strip(),
            }
        else:
            acct["t_trade"] = {"enabled": False}

    base["overall_breakeven"] = float(llm_result.get("overall_breakeven", base["overall_breakeven"]))
    base["recovery_summary"] = str(llm_result.get("recovery_summary", "")).strip()
    base["llm_used"] = True
    logger.info("分账户持仓建议生成完成 — %d 个账户", len(base["accounts"]))
    return base
