"""股票 Profile 运行时切换 — 多股监控时临时覆盖 config.STOCK_*。"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import config


@contextmanager
def ApplyStockProfile(profile: dict[str, Any]) -> Iterator[None]:
    """临时覆盖 config 中的股票身份与持仓，退出后恢复。"""
    saved = {
        "STOCK_CODE": config.STOCK_CODE,
        "STOCK_NAME": config.STOCK_NAME,
        "STOCK_MARKET": config.STOCK_MARKET,
        "STOCK_THEMES": list(config.STOCK_THEMES),
        "STOCK_BUSINESS": config.STOCK_BUSINESS,
        "POSITIONS": list(config.POSITIONS),
    }
    try:
        config.STOCK_CODE = str(profile.get("code", "")).strip()
        config.STOCK_NAME = str(profile.get("name", "")).strip()
        config.STOCK_MARKET = str(profile.get("market", "sz")).strip() or "sz"
        themes = profile.get("themes")
        if isinstance(themes, list):
            config.STOCK_THEMES = [str(t).strip() for t in themes if str(t).strip()]
        else:
            config.STOCK_THEMES = []
        config.STOCK_BUSINESS = str(profile.get("business", "")).strip()
        positions = profile.get("positions")
        if isinstance(positions, list):
            config.POSITIONS = positions
        else:
            config.POSITIONS = []
        yield
    finally:
        config.STOCK_CODE = saved["STOCK_CODE"]
        config.STOCK_NAME = saved["STOCK_NAME"]
        config.STOCK_MARKET = saved["STOCK_MARKET"]
        config.STOCK_THEMES = saved["STOCK_THEMES"]
        config.STOCK_BUSINESS = saved["STOCK_BUSINESS"]
        config.POSITIONS = saved["POSITIONS"]


def FilterProfilesByCode(
    profiles: list[dict[str, Any]],
    stock_code: str | None,
) -> list[dict[str, Any]]:
    """按股票代码筛选 profile 列表。"""
    if not stock_code or not stock_code.strip():
        return profiles
    code = stock_code.strip()
    matched = [p for p in profiles if str(p.get("code", "")).strip() == code]
    return matched
