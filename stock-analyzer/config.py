"""配置管理模块 — 从 .env 读取所有配置项。"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_PATH)


def _ParseBool(value: str | None, default: bool) -> bool:
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in ("true", "1", "yes", "on")


def _ParseInt(value: str | None, default: int) -> int:
    if value is None or value.strip() == "":
        return default
    try:
        return int(value.strip())
    except ValueError:
        return default


def _ParsePositions(raw: str | None) -> list[dict[str, float | int]]:
    if raw is None or raw.strip() == "":
        return []
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    return []


def _ParsePushTimes() -> list[str]:
    times_raw = os.getenv("PUSH_TIMES", "").strip()
    if times_raw:
        return [t.strip() for t in times_raw.split(",") if t.strip()]
    single = os.getenv("PUSH_TIME", "09:00").strip()
    return [single] if single else ["09:00"]


DINGTALK_WEBHOOK: str = os.getenv("DINGTALK_WEBHOOK", "")
DINGTALK_SECRET: str = os.getenv("DINGTALK_SECRET", "")
STOCK_CODE: str = os.getenv("STOCK_CODE", "301075")
STOCK_NAME: str = os.getenv("STOCK_NAME", "多瑞生物")
STOCK_MARKET: str = os.getenv("STOCK_MARKET", "sz")
PUSH_TIME: str = os.getenv("PUSH_TIME", "09:00")
PUSH_TIMES: list[str] = _ParsePushTimes()
MANUAL_RUN: bool = _ParseBool(os.getenv("MANUAL_RUN"), True)
KLINE_DAYS: int = _ParseInt(os.getenv("KLINE_DAYS"), 60)
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# 持仓配置
POSITIONS: list[dict[str, float | int]] = _ParsePositions(os.getenv("POSITIONS"))

# DeepSeek 大模型配置
DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
LLM_ENABLED: bool = _ParseBool(os.getenv("LLM_ENABLED"), True)
