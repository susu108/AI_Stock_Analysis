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


def _ParseFloat(value: str | None, default: float) -> float:
    if value is None or value.strip() == "":
        return default
    try:
        return float(value.strip())
    except ValueError:
        return default


def _ParseCsvList(raw: str | None) -> list[str]:
    if raw is None or raw.strip() == "":
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


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

# 联网搜索（涨跌归因政策资讯补全）
WEB_SEARCH_ENABLED: bool = _ParseBool(os.getenv("WEB_SEARCH_ENABLED"), False)
WEB_SEARCH_PROVIDER: str = os.getenv("WEB_SEARCH_PROVIDER", "tavily").strip().lower()
WEB_SEARCH_API_KEY: str = os.getenv("WEB_SEARCH_API_KEY", "")
STOCK_THEMES: list[str] = _ParseCsvList(os.getenv("STOCK_THEMES"))
WEB_SEARCH_CACHE_HOURS: int = _ParseInt(os.getenv("WEB_SEARCH_CACHE_HOURS"), 4)

# 涨跌归因：涨跌幅低于阈值且无量能异常时输出简版
PRICE_MOVE_MIN_PCT: float = _ParseFloat(os.getenv("PRICE_MOVE_MIN_PCT"), 2.0)

# 网页详细报告（GitHub Pages / jsDelivr CDN）
REPORT_WEB_ENABLED: bool = _ParseBool(os.getenv("REPORT_WEB_ENABLED"), False)
REPORT_WEB_BASE_URL: str = os.getenv("REPORT_WEB_BASE_URL", "").strip().rstrip("/")
REPORT_WEB_CDN: str = os.getenv("REPORT_WEB_CDN", "github_pages").strip().lower()
GITHUB_REPO: str = os.getenv("GITHUB_REPO", "susu108/AI_Stock_Analysis").strip()
GITHUB_BRANCH: str = os.getenv("GITHUB_BRANCH", "main").strip()
REPORT_WEB_GHPROXY: str = os.getenv("REPORT_WEB_GHPROXY", "https://ghfast.top").strip().rstrip("/")
REPORT_WEB_OUTPUT_DIR: str = os.getenv("REPORT_WEB_OUTPUT_DIR", "docs/reports").strip()
REPORT_WEB_RETENTION: int = _ParseInt(os.getenv("REPORT_WEB_RETENTION"), 30)
REPORT_WEB_LOCAL_HINT: bool = _ParseBool(os.getenv("REPORT_WEB_LOCAL_HINT"), True)

_REPO_ROOT = Path(__file__).resolve().parent.parent


def BuildGithubPagesReportBase() -> str:
    """构建 GitHub Pages 报告 URL 前缀（Content-Type 为 text/html）。"""
    parts = GITHUB_REPO.split("/", 1)
    if len(parts) != 2:
        return ""
    owner, repo = parts[0].strip(), parts[1].strip()
    if not owner or not repo:
        return ""
    return f"https://{owner}.github.io/{repo}/reports"


def ResolveReportWebBaseUrl() -> str:
    """解析报告公网 URL 前缀（优先 .env 显式配置，否则按 CDN 类型自动生成）。"""
    explicit = REPORT_WEB_BASE_URL.strip().rstrip("/")
    if explicit:
        return explicit
    if not GITHUB_REPO:
        return ""
    pages_base = BuildGithubPagesReportBase()
    if REPORT_WEB_CDN == "ghproxy" and pages_base:
        # 必须走 GitHub Pages，不能用 raw.githubusercontent（会以 text/plain 返回 HTML 源码）
        return f"{REPORT_WEB_GHPROXY}/{pages_base}"
    if REPORT_WEB_CDN == "statically":
        return (
            f"https://cdn.statically.io/gh/{GITHUB_REPO}/{GITHUB_BRANCH}/docs/reports"
        )
    if REPORT_WEB_CDN == "jsdelivr":
        return f"https://cdn.jsdelivr.net/gh/{GITHUB_REPO}@{GITHUB_BRANCH}/docs/reports"
    if REPORT_WEB_CDN == "github_pages" and pages_base:
        return pages_base
    return pages_base


def ResolveReportWebOutputDir() -> Path:
    """解析网页报告输出目录（相对仓库根）。"""
    configured = Path(REPORT_WEB_OUTPUT_DIR)
    if configured.is_absolute():
        return configured
    return _REPO_ROOT / configured

