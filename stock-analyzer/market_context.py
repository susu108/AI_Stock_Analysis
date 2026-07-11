"""市场时段与预测 Horizon 解析。"""

from __future__ import annotations

from typing import Any


def ResolvePredictionHorizon(session_label: str) -> dict[str, Any]:
    """根据推送时段解析近端预测 horizon。"""
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

    return {
        "session_label": label,
        "is_market_open": False,
        "near_term_label": "明日预测",
        "near_term_horizon": "next_day",
        "near_term_target": "下一交易日",
    }
