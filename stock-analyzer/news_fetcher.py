"""资讯采集 — 个股 / 板块 / 政策 / 宏观 多源实时资讯。"""

from __future__ import annotations

import functools
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from typing import Any, Callable, TypeVar

import akshare as ak
import pandas as pd

import config
from utils import NowStr, SetupLogger

logger = SetupLogger(config.LOG_LEVEL)

F = TypeVar("F", bound=Callable[..., Any])

_CONTENT_MAX_LEN = 300
_CONTENT_MAX_LEN_POLICY = 600
_STOCK_LIMIT = 10
_SECTOR_LIMIT = 16
_POLICY_LIMIT = 10
_MACRO_LIMIT = 5
_CCTV_LOOKBACK_DAYS = 3

_DEFAULT_SECTOR_KEYWORDS = (
    "医药", "医疗", "生物", "制药", "创新药", "疫苗", "医疗器械",
    "授权", "BD", "海外授权", "多肽", "减肥药", "原料药", "连板", "涨停潮",
)
_POLICY_KEYWORDS = (
    "政策", "国务院", "央行", "证监会", "医保", "药监", "监管",
    "发展改革", "财政", "关税", "降息", "降准", "集采", "带量采购",
    "创新", "审批", "注册", "上市", "印发", "通知", "意见", "办法",
    "基药目录", "国谈", "GLP-1", "司美格鲁肽", "替尔泊肽", "公告",
)
_POLICY_EXCLUDE = ("欢迎", "会谈", "访问", "仪式", "会见", "致电", "贺电")


def _Retry(max_retries: int = 3, delay: float = 2.0) -> Callable[[F], F]:
    """重试装饰器。"""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_error: Exception | None = None
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_error = exc
                    logger.warning(
                        "%s 第 %d/%d 次失败: %s",
                        func.__name__,
                        attempt,
                        max_retries,
                        exc,
                    )
                    if attempt < max_retries:
                        time.sleep(delay)
            logger.error("%s 全部重试失败: %s", func.__name__, last_error)
            return None

        return wrapper  # type: ignore[return-value]

    return decorator


def _MakeNewsItem(
    title: str,
    content: str,
    pub_time: str,
    source: str,
    category: str,
    url: str = "",
    content_max_len: int | None = None,
) -> dict[str, str]:
    """构造标准化资讯条目。"""
    max_len = content_max_len if content_max_len is not None else _CONTENT_MAX_LEN
    text = content.strip()
    if len(text) > max_len:
        text = text[:max_len] + "..."
    return {
        "title": title.strip(),
        "content": text,
        "time": pub_time.strip(),
        "source": source.strip(),
        "category": category,
        "url": url.strip(),
    }


def ExtractSectorKeywords(data: dict[str, Any] | None) -> list[str]:
    """从板块数据与默认关键词提取板块检索词。"""
    from sector_catalyst_watch import ExtraSectorKeywords

    keywords: set[str] = set(_DEFAULT_SECTOR_KEYWORDS)
    keywords.update(ExtraSectorKeywords())

    if data:
        for board_key in ("concept", "industry"):
            board = data.get(board_key)
            if board is None or getattr(board, "empty", True):
                continue
            for i in range(min(5, len(board))):
                name = str(board.iloc[i].get("板块名称", "")).strip()
                if len(name) >= 2:
                    keywords.add(name)

    return sorted(keywords)


def BuildSectorSnapshot(data: dict[str, Any] | None) -> list[str]:
    """构建板块涨跌快照文本。"""
    lines: list[str] = []
    if not data:
        return lines

    concept = data.get("concept")
    if concept is not None and not concept.empty:
        lines.append("概念板块TOP5：")
        for i in range(min(5, len(concept))):
            row = concept.iloc[i]
            name = str(row.get("板块名称", ""))
            pct = float(row.get("涨跌幅", 0))
            lines.append(f"  - {name} {pct:+.2f}%")

    industry = data.get("industry")
    if industry is not None and not industry.empty:
        lines.append("行业板块TOP5：")
        for i in range(min(5, len(industry))):
            row = industry.iloc[i]
            name = str(row.get("板块名称", ""))
            pct = float(row.get("涨跌幅", 0))
            lines.append(f"  - {name} {pct:+.2f}%")

    return lines


@_Retry(max_retries=3, delay=2)
def FetchStockNews(limit: int = _STOCK_LIMIT) -> pd.DataFrame | None:
    """获取个股最新资讯。"""
    df = ak.stock_news_em(symbol=config.STOCK_CODE)
    if df is None or df.empty:
        return None
    return df.head(limit).reset_index(drop=True)


@_Retry(max_retries=2, delay=2)
def FetchGlobalNews() -> pd.DataFrame | None:
    """获取东方财富全球财经资讯。"""
    df = ak.stock_info_global_em()
    if df is None or df.empty:
        return None
    return df


@_Retry(max_retries=2, delay=2)
def FetchCctvNews(day: date | None = None) -> pd.DataFrame | None:
    """获取指定日期央视新闻（含政策类）。"""
    target = day or date.today()
    day_str = target.strftime("%Y%m%d")
    df = ak.news_cctv(date=day_str)
    if df is None or df.empty:
        return None
    return df


def FetchCctvNewsMultiDay(days: int = _CCTV_LOOKBACK_DAYS) -> pd.DataFrame | None:
    """回溯多日央视新闻并合并。"""
    frames: list[pd.DataFrame] = []
    today = date.today()
    for offset in range(days):
        target = today - timedelta(days=offset)
        df = FetchCctvNews(target)
        if df is not None and not df.empty:
            copy = df.copy()
            copy["_fetch_date"] = target.strftime("%Y%m%d")
            frames.append(copy)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


@_Retry(max_retries=2, delay=2)
def FetchMacroCalendar() -> pd.DataFrame | None:
    """获取百度经济日历（宏观事件）。"""
    today = date.today().strftime("%Y%m%d")
    df = ak.news_economic_baidu(date=today)
    if df is None or df.empty:
        return None
    return df


def NormalizeStockNews(df: pd.DataFrame | None) -> list[dict[str, str]]:
    """规范化个股资讯。"""
    if df is None or df.empty:
        return []

    items: list[dict[str, str]] = []
    for _, row in df.iterrows():
        title = str(row.get("新闻标题", row.get("title", ""))).strip()
        if not title:
            continue
        items.append(_MakeNewsItem(
            title=title,
            content=str(row.get("新闻内容", row.get("content", ""))),
            pub_time=str(row.get("发布时间", row.get("datetime", ""))),
            source=str(row.get("文章来源", row.get("source", "东方财富"))),
            category="stock",
            url=str(row.get("新闻链接", row.get("链接", ""))),
        ))
    return items


def FilterSectorNews(df: pd.DataFrame | None, keywords: list[str], limit: int) -> list[dict[str, str]]:
    """从全球资讯中筛选板块/行业相关新闻。"""
    if df is None or df.empty or not keywords:
        return []

    pattern = "|".join(re.escape(k) for k in keywords if k)
    if not pattern:
        return []

    title_col = "标题" if "标题" in df.columns else df.columns[0]
    summary_col = "摘要" if "摘要" in df.columns else None
    time_col = "发布时间" if "发布时间" in df.columns else ""
    link_col = "链接" if "链接" in df.columns else ""

    mask = df[title_col].astype(str).str.contains(pattern, na=False, regex=True)
    if summary_col and summary_col in df.columns:
        mask = mask | df[summary_col].astype(str).str.contains(pattern, na=False, regex=True)

    filtered = df[mask].head(limit)
    items: list[dict[str, str]] = []
    for _, row in filtered.iterrows():
        title = str(row.get(title_col, "")).strip()
        if not title:
            continue
        items.append(_MakeNewsItem(
            title=title,
            content=str(row.get(summary_col, "")) if summary_col else "",
            pub_time=str(row.get(time_col, "")),
            source="东方财富",
            category="sector",
            url=str(row.get(link_col, "")),
        ))
    return items


def FilterPolicyNews(
    cctv_df: pd.DataFrame | None,
    global_df: pd.DataFrame | None,
    limit: int,
) -> list[dict[str, str]]:
    """从央视与全球资讯中筛选政策/监管类资讯。"""
    items: list[dict[str, str]] = []

    if cctv_df is not None and not cctv_df.empty:
        pattern = "|".join(re.escape(k) for k in _POLICY_KEYWORDS)
        exclude = "|".join(re.escape(k) for k in _POLICY_EXCLUDE)
        mask = cctv_df["title"].astype(str).str.contains(pattern, na=False, regex=True)
        if "content" in cctv_df.columns:
            mask = mask | cctv_df["content"].astype(str).str.contains(pattern, na=False, regex=True)
        mask = mask & ~cctv_df["title"].astype(str).str.contains(exclude, na=False, regex=True)
        for _, row in cctv_df[mask].head(limit).iterrows():
            title = str(row.get("title", "")).strip()
            if title:
                items.append(_MakeNewsItem(
                    title=title,
                    content=str(row.get("content", "")),
                    pub_time=str(row.get("date", row.get("_fetch_date", ""))),
                    source="央视新闻",
                    category="policy",
                    content_max_len=_CONTENT_MAX_LEN_POLICY,
                ))

    if global_df is not None and not global_df.empty and len(items) < limit:
        title_col = "标题"
        summary_col = "摘要" if "摘要" in global_df.columns else None
        pattern = "|".join(re.escape(k) for k in _POLICY_KEYWORDS)
        mask = global_df[title_col].astype(str).str.contains(pattern, na=False, regex=True)
        if summary_col:
            mask = mask | global_df[summary_col].astype(str).str.contains(pattern, na=False, regex=True)
        for _, row in global_df[mask].head(limit - len(items)).iterrows():
            title = str(row.get(title_col, "")).strip()
            if not title or any(ex in title for ex in _POLICY_EXCLUDE):
                continue
            items.append(_MakeNewsItem(
                title=title,
                content=str(row.get(summary_col, "")) if summary_col else "",
                pub_time=str(row.get("发布时间", "")),
                source="东方财富",
                category="policy",
                url=str(row.get("链接", "")),
                content_max_len=_CONTENT_MAX_LEN_POLICY,
            ))

    return items[:limit]


def NormalizeMacroEvents(df: pd.DataFrame | None, limit: int) -> list[dict[str, str]]:
    """规范化宏观/经济日历事件。"""
    if df is None or df.empty:
        return []

    subset = df.copy()
    if "地区" in subset.columns:
        china_mask = subset["地区"].astype(str).str.contains("中国", na=False)
        if china_mask.any():
            subset = subset[china_mask]

    if "重要性" in subset.columns:
        important = subset[subset["重要性"].astype(float) >= 3]
        if not important.empty:
            subset = important

    items: list[dict[str, str]] = []
    for _, row in subset.head(limit).iterrows():
        event = str(row.get("事件", "")).strip()
        if not event:
            continue
        region = str(row.get("地区", ""))
        pub = row.get("公布", "")
        expect = row.get("预期", "")
        prev = row.get("前值", "")
        event_time = str(row.get("时间", ""))
        content = f"公布:{pub} 预期:{expect} 前值:{prev}".replace("nan", "-")
        items.append(_MakeNewsItem(
            title=f"[{region}] {event}",
            content=content,
            pub_time=event_time,
            source="经济日历",
            category="macro",
        ))
    return items


def FetchNews(data: dict[str, Any] | None = None) -> dict[str, Any]:
    """实时采集多源资讯并分类返回（无缓存，每次调用重新拉取）。"""
    from web_search_fetcher import SearchPolicyAndThemeNews

    fetched_at = NowStr()
    logger.info(
        "开始实时采集 %s 综合资讯（个股/板块/政策/宏观/联网）— %s",
        config.STOCK_NAME,
        fetched_at,
    )

    sector_keywords = ExtractSectorKeywords(data)
    sector_snapshot = BuildSectorSnapshot(data)

    stock_df: pd.DataFrame | None = None
    global_df: pd.DataFrame | None = None
    cctv_df: pd.DataFrame | None = None
    macro_df: pd.DataFrame | None = None

    tasks = {
        "stock": FetchStockNews,
        "global": FetchGlobalNews,
        "cctv": FetchCctvNewsMultiDay,
        "macro": FetchMacroCalendar,
    }
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fn): name for name, fn in tasks.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
                if name == "stock":
                    stock_df = result
                elif name == "global":
                    global_df = result
                elif name == "cctv":
                    cctv_df = result
                elif name == "macro":
                    macro_df = result
            except Exception as exc:
                logger.warning("资讯源 %s 采集异常: %s", name, exc)

    stock_items = NormalizeStockNews(stock_df)
    sector_items = FilterSectorNews(global_df, sector_keywords, _SECTOR_LIMIT)
    from sector_catalyst_watch import (
        CollectCatalystBonus,
        MergeSectorWithCatalystBonus,
        TagCatalystItems,
    )
    catalyst_bonus = CollectCatalystBonus(global_df)
    sector_items = MergeSectorWithCatalystBonus(sector_items, catalyst_bonus)
    sector_items = TagCatalystItems(sector_items)
    policy_items = FilterPolicyNews(cctv_df, global_df, _POLICY_LIMIT)
    macro_items = NormalizeMacroEvents(macro_df, _MACRO_LIMIT)

    web_search_items: list[dict[str, str]] = []
    try:
        web_search_items = SearchPolicyAndThemeNews()
        web_search_items = TagCatalystItems(web_search_items)
    except Exception as exc:
        logger.warning("联网搜索资讯采集异常: %s", exc)

    categories = {
        "stock": stock_items,
        "sector": sector_items,
        "policy": policy_items,
        "macro": macro_items,
        "web_search": web_search_items,
    }
    for cat, items in categories.items():
        if not items and cat != "web_search":
            logger.warning("资讯类别 %s 本次未采集到数据", cat)

    all_items = stock_items + sector_items + policy_items + macro_items + web_search_items
    all_items = TagCatalystItems(all_items)
    catalyst_count = sum(1 for i in all_items if i.get("is_catalyst"))
    bundle: dict[str, Any] = {
        "items": all_items,
        "stock": stock_items,
        "sector": sector_items,
        "policy": policy_items,
        "macro": macro_items,
        "web_search": web_search_items,
        "sector_keywords": sector_keywords,
        "sector_snapshot": sector_snapshot,
        "fetched_at": fetched_at,
        "category_counts": {k: len(v) for k, v in categories.items()},
        "catalyst_count": catalyst_count,
        "is_realtime": True,
    }

    counts = bundle["category_counts"]
    logger.info(
        "实时资讯采集完成 — 个股:%d 板块:%d 政策:%d 宏观:%d 联网:%d "
        "催化:%d 合计:%d ｜ %s",
        counts.get("stock", 0),
        counts.get("sector", 0),
        counts.get("policy", 0),
        counts.get("macro", 0),
        counts.get("web_search", 0),
        catalyst_count,
        len(all_items),
        fetched_at,
    )
    return bundle


def FlattenNewsItems(news_bundle: dict[str, Any] | list[dict[str, str]]) -> list[dict[str, str]]:
    """兼容旧格式，返回扁平资讯列表。"""
    if isinstance(news_bundle, list):
        return news_bundle
    return list(news_bundle.get("items", []))
