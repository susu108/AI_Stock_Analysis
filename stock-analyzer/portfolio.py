"""持仓管理 — 多笔成本、分账户、加权成本、回本计算。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import config
from utils import SetupLogger

logger = SetupLogger(config.LOG_LEVEL)


@dataclass
class PositionLot:
    """单笔持仓。"""

    account: str
    cost: float
    shares: int


@dataclass
class PortfolioSummary:
    """持仓汇总。"""

    lots: list[PositionLot]
    total_shares: int
    weighted_cost: float
    total_cost: float
    market_value: float
    total_pnl: float
    total_pnl_pct: float
    breakeven_price: float
    distance_to_breakeven_pct: float
    t_available_shares: int


def LoadPortfolio() -> list[PositionLot]:
    """从配置加载持仓列表。"""
    lots: list[PositionLot] = []
    for idx, item in enumerate(config.POSITIONS):
        try:
            cost = float(item.get("cost", 0))
            shares = int(item.get("shares", 0))
            account = str(item.get("account", f"账户{idx + 1}")).strip()
            if cost > 0 and shares > 0:
                lots.append(PositionLot(
                    account=account or f"账户{idx + 1}",
                    cost=round(cost, 3),
                    shares=shares,
                ))
        except (TypeError, ValueError) as exc:
            logger.warning("持仓项解析失败: %s — %s", item, exc)
    return lots


def CalcLotSummary(current_price: float, lot: PositionLot) -> dict[str, Any]:
    """计算单账户持仓摘要。"""
    cost_amount = lot.cost * lot.shares
    market_value = current_price * lot.shares
    pnl = market_value - cost_amount
    pnl_pct = (current_price - lot.cost) / lot.cost * 100 if lot.cost > 0 else 0.0
    distance_pct = (lot.cost - current_price) / current_price * 100 if current_price > 0 else 0.0
    return {
        "account": lot.account,
        "cost": lot.cost,
        "shares": lot.shares,
        "cost_amount": round(cost_amount, 2),
        "market_value": round(market_value, 2),
        "pnl": round(pnl, 2),
        "pnl_pct": round(pnl_pct, 2),
        "breakeven_price": lot.cost,
        "distance_to_breakeven_pct": round(distance_pct, 2),
    }


def CalcPortfolioSummary(current_price: float) -> PortfolioSummary | None:
    """计算持仓汇总信息。"""
    lots = LoadPortfolio()
    if not lots or current_price <= 0:
        return None

    total_shares = sum(lot.shares for lot in lots)
    total_cost = sum(lot.cost * lot.shares for lot in lots)
    weighted_cost = total_cost / total_shares if total_shares > 0 else 0.0
    market_value = current_price * total_shares
    total_pnl = market_value - total_cost
    total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0.0
    distance_pct = (
        (weighted_cost - current_price) / current_price * 100
        if current_price > 0
        else 0.0
    )

    return PortfolioSummary(
        lots=lots,
        total_shares=total_shares,
        weighted_cost=round(weighted_cost, 2),
        total_cost=round(total_cost, 2),
        market_value=round(market_value, 2),
        total_pnl=round(total_pnl, 2),
        total_pnl_pct=round(total_pnl_pct, 2),
        breakeven_price=round(weighted_cost, 2),
        distance_to_breakeven_pct=round(distance_pct, 2),
        t_available_shares=total_shares,
    )


def FormatPortfolioText(summary: PortfolioSummary, current_price: float) -> list[str]:
    """格式化持仓文本供 LLM 和报告使用。"""
    lines: list[str] = [
        f"- 现价：{current_price:.2f} 元",
        f"- 总持仓：{summary.total_shares} 股",
        f"- 加权成本：{summary.weighted_cost:.2f} 元",
        f"- 回本目标价：{summary.breakeven_price:.2f} 元（需涨 {summary.distance_to_breakeven_pct:.1f}%）",
        f"- 总市值：{summary.market_value:.2f} 元",
        f"- 浮动盈亏：{summary.total_pnl:+.2f} 元（{summary.total_pnl_pct:+.2f}%）",
        "- 分账户明细：",
    ]
    for lot in summary.lots:
        lot_info = CalcLotSummary(current_price, lot)
        lines.append(
            f"  - {lot.account}：成本 {lot.cost:.3f} × {lot.shares}股 "
            f"→ 浮盈浮亏 {lot_info['pnl']:+.2f}元（{lot_info['pnl_pct']:+.1f}%）"
        )
    return lines


def PortfolioToDict(summary: PortfolioSummary, current_price: float) -> dict[str, Any]:
    """将持仓汇总转为字典。"""
    return {
        "lots": [
            CalcLotSummary(current_price, lot) for lot in summary.lots
        ],
        "total_shares": summary.total_shares,
        "weighted_cost": summary.weighted_cost,
        "total_cost": summary.total_cost,
        "market_value": summary.market_value,
        "total_pnl": summary.total_pnl,
        "total_pnl_pct": summary.total_pnl_pct,
        "breakeven_price": summary.breakeven_price,
        "distance_to_breakeven_pct": summary.distance_to_breakeven_pct,
        "t_available_shares": summary.t_available_shares,
    }
