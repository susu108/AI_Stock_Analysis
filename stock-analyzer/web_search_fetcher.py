"""联网搜索资讯 — Tavily / Serper，补全政策与题材催化。"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from typing import Any

import requests

import config
from utils import NowStr, SetupLogger

logger = SetupLogger(config.LOG_LEVEL)

_SEARCH_LIMIT = 8
_CONTENT_MAX_LEN = 600
_CACHE: dict[str, tuple[float, list[dict[str, str]]]] = {}


class WebSearchProvider(ABC):
    """联网搜索 Provider 抽象基类。"""

    @abstractmethod
    def Search(self, query: str, max_results: int) -> list[dict[str, str]]:
        """执行搜索并返回原始结果列表。"""


class TavilyProvider(WebSearchProvider):
    """Tavily 搜索 Provider。"""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def Search(self, query: str, max_results: int) -> list[dict[str, str]]:
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": self._api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": max_results,
            "include_answer": False,
        }
        try:
            resp = requests.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            body = resp.json()
            results: list[dict[str, str]] = []
            for item in body.get("results", []):
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title", "")).strip()
                if not title:
                    continue
                results.append({
                    "title": title,
                    "content": str(item.get("content", "")).strip(),
                    "url": str(item.get("url", "")).strip(),
                    "source": "Tavily",
                })
            return results
        except requests.Timeout:
            logger.error("Tavily 搜索超时: %s", query)
            return []
        except requests.HTTPError as exc:
            detail = exc.response.text[:200] if exc.response is not None else ""
            logger.error("Tavily HTTP 错误: %s — %s", exc, detail)
            return []
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.error("Tavily 响应解析失败: %s", exc)
            return []
        except Exception as exc:
            logger.error("Tavily 搜索异常: %s — %s", query, exc)
            return []


class SerperProvider(WebSearchProvider):
    """Serper (Google) 搜索 Provider。"""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def Search(self, query: str, max_results: int) -> list[dict[str, str]]:
        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": self._api_key,
            "Content-Type": "application/json",
        }
        payload = {"q": query, "num": max_results}
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            body = resp.json()
            results: list[dict[str, str]] = []
            for item in body.get("organic", []):
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title", "")).strip()
                if not title:
                    continue
                results.append({
                    "title": title,
                    "content": str(item.get("snippet", "")).strip(),
                    "url": str(item.get("link", "")).strip(),
                    "source": "Serper",
                })
            return results
        except requests.Timeout:
            logger.error("Serper 搜索超时: %s", query)
            return []
        except requests.HTTPError as exc:
            detail = exc.response.text[:200] if exc.response is not None else ""
            logger.error("Serper HTTP 错误: %s — %s", exc, detail)
            return []
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.error("Serper 响应解析失败: %s", exc)
            return []
        except Exception as exc:
            logger.error("Serper 搜索异常: %s — %s", query, exc)
            return []


def CreateWebSearchProvider() -> WebSearchProvider | None:
    """根据配置创建搜索 Provider。"""
    if not config.WEB_SEARCH_ENABLED:
        return None
    api_key = config.WEB_SEARCH_API_KEY.strip()
    if not api_key:
        logger.warning("WEB_SEARCH_ENABLED=true 但未配置 WEB_SEARCH_API_KEY")
        return None
    provider_name = config.WEB_SEARCH_PROVIDER
    if provider_name == "serper":
        return SerperProvider(api_key)
    if provider_name == "tavily":
        return TavilyProvider(api_key)
    logger.error("未知 WEB_SEARCH_PROVIDER: %s（支持 tavily / serper）", provider_name)
    return None


def _CacheKey(stock_code: str) -> str:
    return f"{stock_code}:{date.today().isoformat()}"


def _ReadCache(stock_code: str) -> list[dict[str, str]] | None:
    key = _CacheKey(stock_code)
    cached = _CACHE.get(key)
    if cached is None:
        return None
    ts, items = cached
    ttl = max(config.WEB_SEARCH_CACHE_HOURS, 1) * 3600
    if time.time() - ts > ttl:
        _CACHE.pop(key, None)
        return None
    return items


def _WriteCache(stock_code: str, items: list[dict[str, str]]) -> None:
    _CACHE[_CacheKey(stock_code)] = (time.time(), items)


def _NormalizeSearchResult(raw: dict[str, str], query: str) -> dict[str, str]:
    """标准化为 news_fetcher 兼容结构。"""
    content = raw.get("content", "").strip()
    if len(content) > _CONTENT_MAX_LEN:
        content = content[:_CONTENT_MAX_LEN] + "..."
    return {
        "title": raw.get("title", "").strip(),
        "content": content,
        "time": NowStr(),
        "source": raw.get("source", "联网搜索"),
        "category": "web_search",
        "url": raw.get("url", "").strip(),
        "search_query": query,
    }


def BuildSearchQueries(stock_name: str, stock_code: str, themes: list[str]) -> list[str]:
    """构造并行搜索 query 列表。"""
    today = datetime.now()
    yesterday = (today - timedelta(days=1)).strftime("%m月%d日")
    queries: list[str] = [
        f"{stock_name} {stock_code} 上涨 原因 政策",
        f"{stock_name} 板块 资金 行情",
        f"国家基药目录 医保 药监 最新 政策",
    ]
    for theme in themes[:3]:
        queries.append(f"{theme} 政策 A股 医药")
    if yesterday:
        queries.append(f"{stock_name} {yesterday} 大涨 原因")
    return queries[:5]


def _DedupeResults(items: list[dict[str, str]]) -> list[dict[str, str]]:
    """按标题去重。"""
    seen: set[str] = set()
    deduped: list[dict[str, str]] = []
    for item in items:
        key = item.get("title", "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _RunSingleSearch(
    provider: WebSearchProvider,
    query: str,
    per_query: int,
) -> list[dict[str, str]]:
    """执行单次搜索并标准化。"""
    raw_items = provider.Search(query, per_query)
    return [_NormalizeSearchResult(raw, query) for raw in raw_items]


def SearchPolicyAndThemeNews(
    stock_name: str | None = None,
    stock_code: str | None = None,
    themes: list[str] | None = None,
) -> list[dict[str, str]]:
    """搜索政策与题材相关资讯，带缓存。"""
    name = stock_name or config.STOCK_NAME
    code = stock_code or config.STOCK_CODE
    theme_list = themes if themes is not None else config.STOCK_THEMES

    cached = _ReadCache(code)
    if cached is not None:
        logger.info("联网搜索命中缓存 — %d 条", len(cached))
        return cached

    provider = CreateWebSearchProvider()
    if provider is None:
        return []

    queries = BuildSearchQueries(name, code, theme_list)
    per_query = max(2, _SEARCH_LIMIT // max(len(queries), 1))
    collected: list[dict[str, str]] = []

    try:
        with ThreadPoolExecutor(max_workers=min(4, len(queries))) as pool:
            futures = {
                pool.submit(_RunSingleSearch, provider, query, per_query): query
                for query in queries
            }
            for future in as_completed(futures):
                query = futures[future]
                try:
                    collected.extend(future.result())
                except Exception as exc:
                    logger.warning("搜索 query 失败 [%s]: %s", query, exc)
    except Exception as exc:
        logger.error("联网搜索线程池异常: %s", exc)
        return []

    deduped = _DedupeResults(collected)[:_SEARCH_LIMIT]
    _WriteCache(code, deduped)
    logger.info("联网搜索完成 — %d 条（query=%d）", len(deduped), len(queries))
    return deduped
