"""工具函数 — 交易日判断、日志、数值格式化。"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

# 2025-2026 年 A 股休市日（元旦、春节、清明、劳动节、端午、中秋、国庆）
HOLIDAYS: frozenset[date] = frozenset(
    {
        # 2025
        date(2025, 1, 1),
        date(2025, 1, 28),
        date(2025, 1, 29),
        date(2025, 1, 30),
        date(2025, 1, 31),
        date(2025, 2, 1),
        date(2025, 2, 2),
        date(2025, 2, 3),
        date(2025, 2, 4),
        date(2025, 4, 4),
        date(2025, 4, 5),
        date(2025, 4, 6),
        date(2025, 5, 1),
        date(2025, 5, 2),
        date(2025, 5, 3),
        date(2025, 5, 4),
        date(2025, 5, 5),
        date(2025, 5, 31),
        date(2025, 6, 1),
        date(2025, 6, 2),
        date(2025, 10, 1),
        date(2025, 10, 2),
        date(2025, 10, 3),
        date(2025, 10, 4),
        date(2025, 10, 5),
        date(2025, 10, 6),
        date(2025, 10, 7),
        date(2025, 10, 8),
        # 2026
        date(2026, 1, 1),
        date(2026, 1, 2),
        date(2026, 1, 3),
        date(2026, 2, 17),
        date(2026, 2, 18),
        date(2026, 2, 19),
        date(2026, 2, 20),
        date(2026, 2, 21),
        date(2026, 2, 22),
        date(2026, 2, 23),
        date(2026, 4, 4),
        date(2026, 4, 5),
        date(2026, 4, 6),
        date(2026, 5, 1),
        date(2026, 5, 2),
        date(2026, 5, 3),
        date(2026, 6, 19),
        date(2026, 6, 20),
        date(2026, 6, 21),
        date(2026, 9, 25),
        date(2026, 9, 26),
        date(2026, 9, 27),
        date(2026, 10, 1),
        date(2026, 10, 2),
        date(2026, 10, 3),
        date(2026, 10, 4),
        date(2026, 10, 5),
        date(2026, 10, 6),
        date(2026, 10, 7),
    }
)


def IsTradingDay(check_date: date | None = None) -> bool:
    """判断是否为 A 股交易日（排除周末和节假日）。"""
    if check_date is None:
        check_date = date.today()
    if check_date.weekday() >= 5:
        return False
    if check_date in HOLIDAYS:
        return False
    return True


def SetupLogger(level: str = "INFO") -> logging.Logger:
    """配置并返回项目根 logger。"""
    logger = logging.getLogger("stock_analyzer")
    if logger.handlers:
        return logger
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    handler = logging.StreamHandler()
    handler.setLevel(numeric_level)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def SafeFloat(value: Any, default: float = 0.0) -> float:
    """安全转换为 float。"""
    if value is None:
        return default
    try:
        result = float(value)
        if result != result:  # NaN check
            return default
        return result
    except (TypeError, ValueError):
        return default


def FormatVolume(volume: float) -> str:
    """格式化成交量（手 → 万股）。"""
    if volume >= 10000:
        return f"{volume / 10000:.2f}万股"
    return f"{volume:.0f}手"


def FormatAmount(amount: float) -> str:
    """格式化成交额（元 → 亿元/万元）。"""
    if amount >= 1e8:
        return f"{amount / 1e8:.2f}亿元"
    if amount >= 1e4:
        return f"{amount / 1e4:.2f}万元"
    return f"{amount:.0f}元"


def FormatPercent(value: float, with_sign: bool = True) -> str:
    """格式化百分比。"""
    if with_sign and value > 0:
        return f"+{value:.2f}%"
    return f"{value:.2f}%"


def FormatPrice(value: float) -> str:
    """格式化价格。"""
    return f"{value:.2f}"


def NowStr() -> str:
    """返回当前时间字符串。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M")
