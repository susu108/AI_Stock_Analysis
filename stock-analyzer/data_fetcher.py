"""数据采集模块 — 多数据源封装，东方财富不可用时自动降级。"""

from __future__ import annotations

import functools
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from io import StringIO
from typing import Any, Callable, TypeVar

import akshare as ak
import pandas as pd
import requests
from bs4 import BeautifulSoup
from py_mini_racer import MiniRacer

import config
from utils import IsMarketSessionOpen, IsTradingDay, SafeFloat, SetupLogger

logger = SetupLogger(config.LOG_LEVEL)

F = TypeVar("F", bound=Callable[..., Any])

_SINA_HEADERS = {"Referer": "https://finance.sina.com.cn"}
_THS_PAGE_LIMIT = 30
_THS_LOCK = threading.Lock()


def _FullSymbol() -> str:
    """生成带市场前缀的股票代码，如 sz301075。"""
    return f"{config.STOCK_MARKET}{config.STOCK_CODE}"


def _ParsePercent(value: Any) -> float:
    """解析百分比字符串或数值。"""
    if value is None:
        return 0.0
    text = str(value).strip().replace("%", "")
    return SafeFloat(text)


def _ParseChineseAmount(value: Any) -> float:
    """解析中文金额单位（万/亿）为元。"""
    if value is None:
        return 0.0
    text = str(value).strip().replace(",", "")
    if not text or text == "-":
        return 0.0
    try:
        if text.endswith("亿"):
            return float(text[:-1]) * 1e8
        if text.endswith("万"):
            return float(text[:-1]) * 1e4
        return float(text)
    except ValueError:
        return 0.0


def _GetThsColumnValue(row: pd.Series, *must_contain: str) -> Any:
    """按列名关键词模糊匹配取值（兼容「净额(元)」等同花顺列名变体）。"""
    for col in row.index:
        col_str = str(col)
        if all(keyword in col_str for keyword in must_contain):
            return row[col]
    return None


def _GetThsInstantNet(row: pd.Series) -> float:
    """从同花顺「即时」行解析当日净流入。"""
    for col in row.index:
        col_str = str(col)
        if "净额" in col_str and "流入" not in col_str and "流出" not in col_str:
            return _ParseChineseAmount(row[col])
    return 0.0


def _GetThsPeriodNet(row: pd.Series) -> float:
    """从同花顺排行行解析阶段净流入。"""
    for col in row.index:
        col_str = str(col)
        if "资金流入净额" in col_str or "净流入" in col_str:
            return _ParseChineseAmount(row[col])
def _GetThsHeaders() -> dict[str, str]:
    """获取同花顺请求头（含 hexin-v 签名）。"""
    with _THS_LOCK:
        from akshare.datasets import get_ths_js

        js_code = MiniRacer()
        js_path = get_ths_js("ths.js")
        with open(js_path, encoding="utf-8") as f:
            js_code.eval(f.read())
        v_code = js_code.call("v")
    return {
        "Accept": "text/html, */*; q=0.01",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "hexin-v": v_code,
        "Referer": "http://data.10jqka.com.cn/funds/ggzjl/",
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "X-Requested-With": "XMLHttpRequest",
    }


def _Retry(max_retries: int = 3, delay: float = 2.0) -> Callable[[F], F]:
    """重试装饰器。"""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_error: Exception | None = None
            for attempt in range(1, max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    time.sleep(0.3)
                    return result
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


def _NormalizeSinaKline(df: pd.DataFrame) -> pd.DataFrame:
    """将新浪 K 线格式统一为分析模块所需列名。"""
    result = df.rename(columns={
        "date": "日期",
        "open": "开盘",
        "close": "收盘",
        "high": "最高",
        "low": "最低",
        "volume": "成交量",
        "amount": "成交额",
        "turnover": "换手率",
    }).copy()

    close = result["收盘"].astype(float)
    prev_close = close.shift(1)
    result["涨跌幅"] = (close.pct_change() * 100).round(2)
    result["涨跌额"] = close.diff().round(2)
    result["振幅"] = (
        (result["最高"].astype(float) - result["最低"].astype(float))
        / prev_close.replace(0, pd.NA)
        * 100
    ).round(2)
    result["日期"] = pd.to_datetime(result["日期"]).dt.strftime("%Y-%m-%d")
    return result


def _FetchSinaRealtime() -> dict[str, Any] | None:
    """新浪单股实时行情（轻量，无需拉全市场）。"""
    symbol = _FullSymbol()
    url = f"https://hq.sinajs.cn/list={symbol}"
    resp = requests.get(url, headers=_SINA_HEADERS, timeout=10)
    resp.encoding = "gbk"
    match = re.search(r'"([^"]+)"', resp.text)
    if not match:
        return None

    parts = match.group(1).split(",")
    if len(parts) < 10 or not parts[3]:
        return None

    name = parts[0]
    open_price = SafeFloat(parts[1])
    prev_close = SafeFloat(parts[2])
    price = SafeFloat(parts[3])
    high = SafeFloat(parts[4])
    low = SafeFloat(parts[5])
    volume = SafeFloat(parts[8])
    amount = SafeFloat(parts[9])
    change_amt = price - prev_close
    change_pct = (change_amt / prev_close * 100) if prev_close else 0.0
    amplitude = ((high - low) / prev_close * 100) if prev_close else 0.0

    return {
        "code": config.STOCK_CODE,
        "name": name or config.STOCK_NAME,
        "price": price,
        "change_pct": round(change_pct, 2),
        "change_amt": round(change_amt, 2),
        "open": open_price,
        "high": high,
        "low": low,
        "prev_close": prev_close,
        "volume": volume,
        "amount": amount,
        "turnover": 0.0,
        "volume_ratio": 0.0,
        "pe": 0.0,
        "pb": 0.0,
        "amplitude": round(amplitude, 2),
        "total_mv": 0.0,
        "circ_mv": 0.0,
    }


def _FetchSinaExtendedInfo() -> dict[str, float]:
    """新浪扩展行情（市盈率、市净率、流通股本等）。"""
    symbol = _FullSymbol()
    result: dict[str, float] = {"pe": 0.0, "pb": 0.0, "total_mv": 0.0, "circ_mv": 0.0}

    try:
        resp = requests.get(
            f"https://hq.sinajs.cn/list={symbol}_i",
            headers=_SINA_HEADERS,
            timeout=10,
        )
        resp.encoding = "gbk"
        match = re.search(r'"([^"]+)"', resp.text)
        if match:
            parts = match.group(1).split(",")
            if len(parts) > 20:
                result["pe"] = SafeFloat(parts[19])
                result["pb"] = SafeFloat(parts[20])
    except Exception as exc:
        logger.warning("新浪扩展行情获取失败: %s", exc)

    try:
        resp = requests.get(
            f"https://finance.sina.com.cn/realstock/company/{symbol}/jsvar.js",
            headers=_SINA_HEADERS,
            timeout=10,
        )
        resp.encoding = "gbk"
        cap_match = re.search(r"var totalcapital = ([\d.]+)", resp.text)
        if cap_match:
            total_shares = SafeFloat(cap_match.group(1))
            price_resp = requests.get(
                f"https://hq.sinajs.cn/list={symbol}",
                headers=_SINA_HEADERS,
                timeout=10,
            )
            price_resp.encoding = "gbk"
            price_match = re.search(r'"([^"]+)"', price_resp.text)
            if price_match and total_shares > 0:
                price = SafeFloat(price_match.group(1).split(",")[3])
                # totalcapital 单位为万股
                result["total_mv"] = total_shares * 10000 * price
                result["circ_mv"] = result["total_mv"]
    except Exception as exc:
        logger.warning("新浪股本信息获取失败: %s", exc)

    return result


def _FetchThsTurnover() -> float:
    """从同花顺即时资金流获取换手率。"""
    try:
        row = _SearchThsStockRow("即时")
        if row is None:
            return 0.0
        for col in row.index:
            if "换手" in str(col):
                return _ParsePercent(row[col])
    except Exception as exc:
        logger.warning("同花顺换手率获取失败: %s", exc)
    return 0.0


def _EnrichRealtimeQuote(
    realtime: dict[str, Any] | None,
    kline: pd.DataFrame | None,
) -> dict[str, Any] | None:
    """用 K 线、新浪扩展接口、同花顺补全换手率/量比/市盈率/市净率。"""
    if realtime is None:
        return None

    if kline is not None and not kline.empty:
        volumes = kline["成交量"].astype(float)
        today_volume = SafeFloat(realtime.get("volume")) or SafeFloat(volumes.iloc[-1])
        if len(volumes) >= 11:
            avg_volume = volumes.iloc[-11:-1].mean()
            if avg_volume > 0:
                realtime["volume_ratio"] = round(today_volume / avg_volume, 2)

        if "换手率" in kline.columns:
            turnover_raw = SafeFloat(kline.iloc[-1]["换手率"])
            # 新浪 K 线换手率为小数（0.032 → 3.2%）
            realtime["turnover"] = round(
                turnover_raw * 100 if turnover_raw < 1 else turnover_raw,
                2,
            )

    if SafeFloat(realtime.get("turnover")) == 0.0:
        ths_turnover = _FetchThsTurnover()
        if ths_turnover > 0:
            realtime["turnover"] = round(ths_turnover, 2)

    ext_info = _FetchSinaExtendedInfo()
    if ext_info["pe"] > 0:
        realtime["pe"] = ext_info["pe"]
    if ext_info["pb"] > 0:
        realtime["pb"] = ext_info["pb"]
    if ext_info["total_mv"] > 0:
        realtime["total_mv"] = ext_info["total_mv"]
        realtime["circ_mv"] = ext_info["circ_mv"]

    return realtime


def _KlineDateStr(row_date: Any) -> str:
    """将 K 线日期列统一为 YYYY-MM-DD 字符串。"""
    if hasattr(row_date, "strftime"):
        return row_date.strftime("%Y-%m-%d")
    text = str(row_date).strip()[:10]
    return text.replace("/", "-")


def MergeRealtimeIntoKline(
    kline: pd.DataFrame | None,
    realtime: dict[str, Any] | None,
    check_time: datetime | None = None,
) -> pd.DataFrame | None:
    """盘中将实时价融合进日 K，保证技术指标与现价一致。"""
    if kline is None or kline.empty or realtime is None:
        return kline
    if not IsMarketSessionOpen(check_time):
        return kline

    price = SafeFloat(realtime.get("price"))
    if price <= 0:
        return kline

    today_str = (check_time or datetime.now()).date().strftime("%Y-%m-%d")
    prev_close = SafeFloat(realtime.get("prev_close"))
    open_price = SafeFloat(realtime.get("open")) or price
    high = max(SafeFloat(realtime.get("high")), price, open_price)
    low_raw = SafeFloat(realtime.get("low"))
    low = min(low_raw, price, open_price) if low_raw > 0 else min(price, open_price)
    volume = SafeFloat(realtime.get("volume"))
    amount = SafeFloat(realtime.get("amount"))
    change_amt = round(price - prev_close, 2) if prev_close else 0.0
    change_pct = round(change_amt / prev_close * 100, 2) if prev_close else 0.0
    amplitude = round((high - low) / prev_close * 100, 2) if prev_close else 0.0

    result = kline.copy()
    last_date = _KlineDateStr(result.iloc[-1]["日期"])

    row_data: dict[str, Any] = {
        "日期": today_str,
        "开盘": open_price,
        "收盘": price,
        "最高": high,
        "最低": low,
        "成交量": volume,
        "成交额": amount,
        "涨跌幅": change_pct,
        "涨跌额": change_amt,
        "振幅": amplitude,
    }
    if "换手率" in result.columns:
        turnover = SafeFloat(realtime.get("turnover"))
        row_data["换手率"] = round(turnover / 100, 4) if turnover > 1 else turnover

    if last_date == today_str:
        for key, value in row_data.items():
            if key in result.columns:
                result.at[result.index[-1], key] = value
        logger.info("K线已融合今日实时价: %.2f（更新）", price)
    elif last_date < today_str:
        result = pd.concat([result, pd.DataFrame([row_data])], ignore_index=True)
        logger.info("K线已融合今日实时价: %.2f（追加）", price)
    else:
        logger.debug("K线末行日期 %s 晚于今日，跳过融合", last_date)

    return result


def _FetchEastmoneyRealtime() -> dict[str, Any] | None:
    """东方财富全市场 spot 中提取单股实时行情。"""
    try:
        df = ak.stock_zh_a_spot_em()
        row = df[df["代码"] == config.STOCK_CODE]
        if row.empty:
            return None
        r = row.iloc[0]
        return {
            "code": str(r["代码"]),
            "name": str(r["名称"]),
            "price": SafeFloat(r["最新价"]),
            "change_pct": SafeFloat(r["涨跌幅"]),
            "change_amt": SafeFloat(r["涨跌额"]),
            "open": SafeFloat(r["今开"]),
            "high": SafeFloat(r["最高"]),
            "low": SafeFloat(r["最低"]),
            "prev_close": SafeFloat(r["昨收"]),
            "volume": SafeFloat(r["成交量"]),
            "amount": SafeFloat(r["成交额"]),
            "turnover": SafeFloat(r["换手率"]),
            "volume_ratio": SafeFloat(r["量比"]),
            "pe": SafeFloat(r.get("市盈率-动态", 0)),
            "pb": SafeFloat(r.get("市净率", 0)),
            "amplitude": SafeFloat(r.get("振幅", 0)),
            "total_mv": SafeFloat(r.get("总市值", 0)),
            "circ_mv": SafeFloat(r.get("流通市值", 0)),
            "source": "eastmoney",
        }
    except Exception as exc:
        logger.warning("东方财富实时行情失败: %s", exc)
        return None


def _PickRealtimeQuote(
    sina: dict[str, Any] | None,
    eastmoney: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """盘中双源校验，优先东财；非盘中优先新浪。"""
    session_open = IsMarketSessionOpen()

    if sina is not None:
        sina = {**sina, "source": "sina"}
    if eastmoney is not None:
        eastmoney = {**eastmoney, "source": "eastmoney"}

    sina_price = SafeFloat(sina.get("price")) if sina else 0.0
    em_price = SafeFloat(eastmoney.get("price")) if eastmoney else 0.0

    if session_open:
        if em_price > 0 and sina_price <= 0:
            logger.info("实时行情采用东财（新浪无效）: %.2f", em_price)
            return eastmoney
        if sina_price > 0 and em_price > 0:
            diff_pct = abs(sina_price - em_price) / em_price * 100
            if diff_pct > 0.5:
                logger.warning(
                    "新浪(%.2f)与东财(%.2f)偏差 %.2f%%，采用东财",
                    sina_price, em_price, diff_pct,
                )
                return eastmoney
            logger.info("实时行情采用新浪: %.2f（东财 %.2f）", sina_price, em_price)
            return sina
        if em_price > 0:
            logger.info("实时行情采用东财: %.2f", em_price)
            return eastmoney
        if sina_price > 0:
            logger.info("实时行情采用新浪: %.2f", sina_price)
            return sina
        return None

    if sina_price > 0:
        logger.info("实时行情采用新浪: %.2f", sina_price)
        return sina
    if em_price > 0:
        logger.info("实时行情采用东财: %.2f", em_price)
        return eastmoney
    return None


def _FetchSinaKline(days: int) -> pd.DataFrame | None:
    """新浪 K 线（前复权）。"""
    end_date = date.today().strftime("%Y%m%d")
    start_date = (date.today() - timedelta(days=days + 30)).strftime("%Y%m%d")
    df = ak.stock_zh_a_daily(
        symbol=_FullSymbol(),
        start_date=start_date,
        end_date=end_date,
        adjust="qfq",
    )
    if df is None or df.empty:
        return None
    normalized = _NormalizeSinaKline(df)
    return normalized.tail(days).reset_index(drop=True)


def _SearchThsStockRow(ranking: str) -> pd.Series | None:
    """在同花顺资金流排行中搜索目标股票（找到即停止，避免拉全量）。"""
    ranking_urls = {
        "即时": "http://data.10jqka.com.cn/funds/ggzjl/field/zdf/order/desc/page/{}/ajax/1/free/1/",
        "3日排行": "http://data.10jqka.com.cn/funds/ggzjl/board/3/field/zdf/order/desc/page/{}/ajax/1/free/1/",
        "5日排行": "http://data.10jqka.com.cn/funds/ggzjl/board/5/field/zdf/order/desc/page/{}/ajax/1/free/1/",
    }
    url_template = ranking_urls.get(ranking)
    if url_template is None:
        return None

    for page in range(1, _THS_PAGE_LIMIT + 1):
        headers = _GetThsHeaders()
        resp = requests.get(url_template.format(page), headers=headers, timeout=15)
        tables = pd.read_html(StringIO(resp.text))
        if not tables:
            break
        page_df = tables[0]
        if "股票代码" not in page_df.columns:
            break
        matched = page_df[page_df["股票代码"].astype(str) == config.STOCK_CODE]
        if not matched.empty:
            return matched.iloc[0]
        if len(page_df) < 50:
            break
    return None


def _BuildThsFundFlow() -> pd.DataFrame | None:
    """基于同花顺数据构建近 5 日资金流向（东方财富不可用时的降级方案）。"""
    instant_row = _SearchThsStockRow("即时")
    if instant_row is None:
        return None

    today_net = _GetThsInstantNet(instant_row)
    five_row = _SearchThsStockRow("5日排行")
    five_total = _GetThsPeriodNet(five_row) if five_row is not None else today_net * 5

    if today_net == 0.0:
        logger.warning("同花顺净额解析为 0，列名: %s", list(instant_row.index))
        return None

    # 5 日累计小于当日时，前 4 日均分剩余部分（可能为负，属正常估算）
    prev_avg = (five_total - today_net) / 4

    rows: list[dict[str, Any]] = []
    for offset in range(4, 0, -1):
        day = date.today() - timedelta(days=offset)
        rows.append({
            "日期": day,
            "主力净流入-净额": prev_avg,
            "超大单净流入-净额": prev_avg * 0.6,
        })
    rows.append({
        "日期": date.today(),
        "主力净流入-净额": today_net,
        "超大单净流入-净额": today_net * 0.6,
    })
    logger.info("资金流向使用同花顺数据源（近5日为估算值）")
    return pd.DataFrame(rows)


def _NormalizeThsBoards(df: pd.DataFrame) -> pd.DataFrame:
    """将同花顺板块数据统一为 板块名称/涨跌幅 格式。"""
    result = df.copy()
    result["板块名称"] = result["行业"]
    result["涨跌幅"] = result["行业-涨跌幅"].apply(_ParsePercent)
    return result.sort_values("涨跌幅", ascending=False).reset_index(drop=True)


@_Retry(max_retries=3, delay=2)
def GetRealtimeQuote() -> dict[str, Any] | None:
    """获取实时行情（新浪 + 东财双源，盘中校验后择优）。"""
    sina_data: dict[str, Any] | None = None
    try:
        sina_data = _FetchSinaRealtime()
    except Exception as exc:
        logger.warning("新浪实时行情失败: %s", exc)

    eastmoney_data: dict[str, Any] | None = None
    if IsMarketSessionOpen() or sina_data is None or SafeFloat(sina_data.get("price")) <= 0:
        eastmoney_data = _FetchEastmoneyRealtime()

    picked = _PickRealtimeQuote(sina_data, eastmoney_data)
    if picked is not None:
        return picked

    if not IsMarketSessionOpen():
        try:
            eastmoney_data = eastmoney_data or _FetchEastmoneyRealtime()
            picked = _PickRealtimeQuote(sina_data, eastmoney_data)
            if picked is not None:
                return picked
        except Exception as exc:
            logger.warning("东财降级失败: %s", exc)

    raise ConnectionError("所有实时行情数据源均失败")


@_Retry(max_retries=3, delay=2)
def GetKline(days: int | None = None) -> pd.DataFrame | None:
    """获取 K 线（优先新浪，降级东方财富）。"""
    if days is None:
        days = config.KLINE_DAYS

    try:
        df = _FetchSinaKline(days)
        if df is not None and not df.empty:
            logger.debug("K线：新浪")
            return df
    except Exception as exc:
        logger.warning("新浪 K 线失败: %s", exc)

    end_date = date.today().strftime("%Y%m%d")
    start_date = (date.today() - timedelta(days=days + 30)).strftime("%Y%m%d")
    df = ak.stock_zh_a_hist(
        symbol=config.STOCK_CODE,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust="qfq",
    )
    if df is None or df.empty:
        return None
    logger.debug("K线：东方财富")
    return df.tail(days).reset_index(drop=True)


@_Retry(max_retries=3, delay=2)
def GetConceptBoards() -> pd.DataFrame | None:
    """获取概念板块（优先同花顺，降级东方财富）。"""
    try:
        df = ak.stock_fund_flow_concept(symbol="即时")
        if df is not None and not df.empty:
            logger.debug("概念板块：同花顺")
            return _NormalizeThsBoards(df)
    except Exception as exc:
        logger.warning("同花顺概念板块失败: %s", exc)

    df = ak.stock_board_concept_name_em()
    if df is None or df.empty:
        return None
    if "涨跌幅" in df.columns:
        df = df.sort_values("涨跌幅", ascending=False).reset_index(drop=True)
    logger.debug("概念板块：东方财富")
    return df


@_Retry(max_retries=3, delay=2)
def GetIndustryBoards() -> pd.DataFrame | None:
    """获取行业板块（优先同花顺，降级东方财富）。"""
    try:
        df = ak.stock_fund_flow_industry(symbol="即时")
        if df is not None and not df.empty:
            logger.debug("行业板块：同花顺")
            return _NormalizeThsBoards(df)
    except Exception as exc:
        logger.warning("同花顺行业板块失败: %s", exc)

    df = ak.stock_board_industry_name_em()
    if df is None or df.empty:
        return None
    if "涨跌幅" in df.columns:
        df = df.sort_values("涨跌幅", ascending=False).reset_index(drop=True)
    logger.debug("行业板块：东方财富")
    return df


@_Retry(max_retries=3, delay=2)
def GetFundFlow() -> pd.DataFrame | None:
    """获取个股资金流向（优先东方财富，降级同花顺）。"""
    try:
        df = ak.stock_individual_fund_flow(
            stock=config.STOCK_CODE,
            market=config.STOCK_MARKET,
        )
        if df is not None and not df.empty:
            logger.debug("资金流向：东方财富")
            return df
    except Exception as exc:
        logger.warning("东方财富资金流向失败: %s", exc)

    return _BuildThsFundFlow()


@_Retry(max_retries=3, delay=2)
def GetLhb(days: int = 5) -> pd.DataFrame | None:
    """获取龙虎榜明细（筛选目标股票）。"""
    end = date.today()
    start = end - timedelta(days=days)
    df = ak.stock_lhb_detail_em(
        start_date=start.strftime("%Y%m%d"),
        end_date=end.strftime("%Y%m%d"),
    )
    if df is None or df.empty:
        return None
    code_col = "代码" if "代码" in df.columns else None
    if code_col is None:
        return None
    filtered = df[df[code_col] == config.STOCK_CODE]
    if filtered.empty:
        return None
    return filtered.reset_index(drop=True)


@_Retry(max_retries=3, delay=2)
def GetMarginDetail() -> dict[str, Any] | None:
    """获取个股融资融券近端数据。"""
    try:
        df = ak.stock_margin_detail_em(symbol=config.STOCK_CODE)
        if df is None or df.empty:
            return None
        tail = df.tail(5).reset_index(drop=True)
        latest = tail.iloc[-1]
        prev = tail.iloc[-2] if len(tail) >= 2 else latest

        balance_col = next(
            (c for c in df.columns if "融资余额" in str(c)),
            None,
        )
        buy_col = next(
            (c for c in df.columns if "融资买入" in str(c)),
            None,
        )
        if balance_col is None:
            return None

        latest_balance = _ParseChineseAmount(latest.get(balance_col))
        prev_balance = _ParseChineseAmount(prev.get(balance_col))
        change = latest_balance - prev_balance if prev_balance > 0 else 0.0
        latest_buy = _ParseChineseAmount(latest.get(buy_col)) if buy_col else 0.0

        date_col = next((c for c in df.columns if "日期" in str(c)), None)
        trade_date = str(latest.get(date_col, "")) if date_col else ""

        return {
            "margin_balance": latest_balance,
            "margin_balance_prev": prev_balance,
            "margin_balance_change": change,
            "margin_buy": latest_buy,
            "trade_date": trade_date,
        }
    except Exception as exc:
        logger.warning("融资融券数据采集失败: %s", exc)
        return None


def FetchAllData() -> dict[str, Any]:
    """采集所有数据：新浪/龙虎榜并行，同花顺相关串行（避免 MiniRacer 线程冲突）。"""
    logger.info("开始采集 %s(%s) 数据...", config.STOCK_NAME, config.STOCK_CODE)

    data: dict[str, Any] = {}
    parallel_fetchers: dict[str, Callable[[], Any]] = {
        "realtime": GetRealtimeQuote,
        "kline": GetKline,
        "lhb": GetLhb,
    }
    serial_fetchers: dict[str, Callable[[], Any]] = {
        "concept": GetConceptBoards,
        "industry": GetIndustryBoards,
        "fund_flow": GetFundFlow,
        "margin": GetMarginDetail,
    }

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_map = {executor.submit(fn): key for key, fn in parallel_fetchers.items()}
        for future in as_completed(future_map):
            key = future_map[future]
            try:
                data[key] = future.result()
            except Exception as exc:
                logger.error("采集 %s 异常: %s", key, exc)
                data[key] = None

    for key, fn in serial_fetchers.items():
        try:
            data[key] = fn()
        except Exception as exc:
            logger.error("采集 %s 异常: %s", key, exc)
            data[key] = None

    data["realtime"] = _EnrichRealtimeQuote(
        data.get("realtime"),
        data.get("kline"),
    )
    data["kline"] = MergeRealtimeIntoKline(
        data.get("kline"),
        data.get("realtime"),
    )
    if data.get("realtime") and data.get("kline") is not None:
        data["realtime"] = _EnrichRealtimeQuote(
            data.get("realtime"),
            data.get("kline"),
        )

    success_count = sum(1 for v in data.values() if v is not None)
    logger.info("数据采集完成，成功 %d/7 项", success_count)
    return data
