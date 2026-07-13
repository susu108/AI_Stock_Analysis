"""市场时段与预测 Horizon 解析。"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from utils import IsTradingDay

_WEEKDAY_CN = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")


def GetNextTradingDay(from_date: date | None = None) -> date:
    """返回 from_date 之后的下一 A 股交易日（不含 from_date 本身）。"""
    cursor = (from_date or date.today()) + timedelta(days=1)
    for _ in range(366):
        if IsTradingDay(cursor):
            return cursor
        cursor += timedelta(days=1)
    raise RuntimeError("无法在 366 天内找到下一交易日")


def FormatTradingDayLabel(trading_day: date) -> str:
    """格式化交易日标签，如 7月13日（周一）。"""
    weekday = _WEEKDAY_CN[trading_day.weekday()]
    return f"{trading_day.month}月{trading_day.day}日（{weekday}）"


def FormatNearTermTarget(next_day: date) -> str:
    """格式化近端预测目标文案。"""
    return f"下一交易日（{FormatTradingDayLabel(next_day)}）"


def ResolvePredictionHorizon(session_label: str) -> dict[str, Any]:
    """根据推送时段解析近端预测 horizon（不含交易日校正）。"""
    label = session_label.strip() or "盘中"

    if label in ("开盘前", "收盘后"):
        return {
            "session_label": label,
            "is_market_open": False,
            "near_term_label": "明日预测",
            "near_term_horizon": "next_day",
            "near_term_target": "下一交易日",
        }

    if label in ("上午盘中", "下午盘中"):
        return {
            "session_label": label,
            "is_market_open": True,
            "near_term_label": "今日预测",
            "near_term_horizon": "today",
            "near_term_target": "今日剩余时段至收盘",
        }

    if label == "午盘":
        return {
            "session_label": label,
            "is_market_open": False,
            "near_term_label": "今日下午预测",
            "near_term_horizon": "today",
            "near_term_target": "今日下午盘",
        }

    if label == "尾盘":
        return {
            "session_label": label,
            "is_market_open": True,
            "near_term_label": "今日预测",
            "near_term_horizon": "today",
            "near_term_target": "今日剩余时段至收盘",
        }

    return {
        "session_label": label,
        "is_market_open": False,
        "near_term_label": "明日预测",
        "near_term_horizon": "next_day",
        "near_term_target": "下一交易日",
    }


def _EnrichHorizonWithTradingDay(
    horizon: dict[str, Any],
    check_date: date,
    is_trading_day: bool,
) -> dict[str, Any]:
    """为 horizon 补充交易日相关字段，并校正 next_day 目标文案。"""
    next_day = GetNextTradingDay(check_date)
    enriched = {
        **horizon,
        "is_trading_day": is_trading_day,
        "next_trading_date": next_day.isoformat(),
    }
    if horizon.get("near_term_horizon") in ("next_day", "next_trading_day"):
        enriched["near_term_target"] = FormatNearTermTarget(next_day)
    return enriched


def ResolveSessionAndHorizon(
    session_label: str,
    check_date: date | None = None,
) -> tuple[str, dict[str, Any]]:
    """解析展示用时段标签与近端预测 horizon（含非交易日校正）。"""
    today = check_date or date.today()
    reference = session_label.strip() or "盘中"
    next_day = GetNextTradingDay(today)

    if not IsTradingDay(today):
        if today.weekday() >= 5:
            display_label = "周末休市"
        else:
            display_label = "节假日休市"
        horizon: dict[str, Any] = {
            "session_label": display_label,
            "reference_session": reference,
            "is_market_open": False,
            "is_trading_day": False,
            "near_term_label": "下交易日预测",
            "near_term_horizon": "next_trading_day",
            "near_term_target": FormatNearTermTarget(next_day),
            "next_trading_date": next_day.isoformat(),
        }
        return display_label, horizon

    base = ResolvePredictionHorizon(reference)
    horizon = _EnrichHorizonWithTradingDay(base, today, is_trading_day=True)
    horizon["reference_session"] = reference
    if base.get("near_term_horizon") == "next_day":
        horizon["near_term_label"] = "下交易日预测"
    return reference, horizon
