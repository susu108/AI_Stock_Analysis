"""历史价位上下文 — 供 AI 参考定价。"""

from __future__ import annotations

from typing import Any

import pandas as pd

import config
from portfolio import CalcPortfolioSummary
from utils import SafeFloat, SetupLogger

logger = SetupLogger(config.LOG_LEVEL)


def ExtractRecentPriceLevels(
    kline: pd.DataFrame | None,
    realtime: dict[str, Any] | None = None,
) -> dict[str, float]:
    """从 K 线与实时行情提取近期真实高低点（含今日/昨日最低）。"""
    levels: dict[str, float] = {}
    rt = realtime or {}

    if kline is not None and not kline.empty:
        low_col = "最低" if "最低" in kline.columns else "low"
        high_col = "最高" if "最高" in kline.columns else "high"
        open_col = "开盘" if "开盘" in kline.columns else "open"
        close_col = "收盘" if "收盘" in kline.columns else "close"

        lows = kline[low_col].astype(float)
        highs = kline[high_col].astype(float)
        opens = kline[open_col].astype(float)
        closes = kline[close_col].astype(float)

        levels["today_low_kline"] = float(lows.iloc[-1])
        levels["today_high_kline"] = float(highs.iloc[-1])
        levels["today_open_kline"] = float(opens.iloc[-1])
        levels["today_close_kline"] = float(closes.iloc[-1])
        if len(lows) >= 2:
            levels["prev_day_low"] = float(lows.iloc[-2])
            levels["prev_day_high"] = float(highs.iloc[-2])
            levels["prev_day_open"] = float(opens.iloc[-2])
            levels["prev_day_close"] = float(closes.iloc[-2])
        levels["low_3"] = float(lows.tail(3).min())
        levels["low_5"] = float(lows.tail(5).min())
        levels["high_3"] = float(highs.tail(3).max())
        levels["high_5"] = float(highs.tail(5).max())

    if rt.get("low"):
        levels["today_low"] = SafeFloat(rt["low"])
    elif levels.get("today_low_kline"):
        levels["today_low"] = levels["today_low_kline"]

    if rt.get("high"):
        levels["today_high"] = SafeFloat(rt["high"])
    elif levels.get("today_high_kline"):
        levels["today_high"] = levels["today_high_kline"]

    if rt.get("open"):
        levels["today_open"] = SafeFloat(rt["open"])
    elif levels.get("today_open_kline"):
        levels["today_open"] = levels["today_open_kline"]

    if rt.get("prev_close"):
        levels["prev_close"] = SafeFloat(rt["prev_close"])
    elif levels.get("prev_day_close"):
        levels["prev_close"] = levels["prev_day_close"]

    return levels


def CalcPriceAnchors(
    kline: pd.DataFrame | None,
    realtime: dict[str, Any] | None,
    indicators: dict[str, Any],
    current_price: float,
) -> dict[str, float]:
    """计算分档定价技术锚点（箱体/VWAP/关键支撑/禁加线等）。"""
    anchors: dict[str, float] = {}
    levels = ExtractRecentPriceLevels(kline, realtime)
    anchors.update(levels)

    ma5 = SafeFloat(indicators.get("ma5"))
    ma20 = SafeFloat(indicators.get("ma20"))

    if kline is not None and not kline.empty:
        low_col = "最低" if "最低" in kline.columns else "low"
        high_col = "最高" if "最高" in kline.columns else "high"
        close_col = "收盘" if "收盘" in kline.columns else "close"
        vol_col = "成交量" if "成交量" in kline.columns else "volume"
        amt_col = "成交额" if "成交额" in kline.columns else "amount"

        lows = kline[low_col].astype(float)
        highs = kline[high_col].astype(float)
        closes = kline[close_col].astype(float)

        box_lows = lows.tail(10)
        box_highs = highs.tail(10)
        anchors["box_low"] = float(box_lows.min())
        anchors["box_high"] = float(box_highs.max())
        anchors["box_mid"] = round(
            (anchors["box_low"] + anchors["box_high"]) / 2, 2,
        )

        if len(kline) >= 2 and amt_col in kline.columns and vol_col in kline.columns:
            prev_vol = SafeFloat(kline[vol_col].iloc[-2])
            prev_amt = SafeFloat(kline[amt_col].iloc[-2])
            if prev_vol > 0:
                anchors["prev_day_vwap"] = round(prev_amt / prev_vol, 2)
            elif len(kline) >= 2:
                prev_h = float(highs.iloc[-2])
                prev_l = float(lows.iloc[-2])
                prev_c = float(closes.iloc[-2])
                anchors["prev_day_vwap"] = round((prev_h + prev_l + prev_c) / 3, 2)

        rally_start = 0.0
        for i in range(len(closes) - 1, max(0, len(closes) - 8), -1):
            if i <= 0:
                break
            day_chg = (float(closes.iloc[i]) - float(closes.iloc[i - 1])) / float(closes.iloc[i - 1])
            if day_chg >= 0.03:
                rally_start = float(lows.iloc[i - 1])
                break
        if rally_start <= 0 and len(lows) >= 3:
            rally_start = float(lows.iloc[-3])
        anchors["rally_start_low"] = rally_start

    if anchors.get("prev_day_vwap", 0) <= 0 and ma5 > 0:
        anchors["prev_day_vwap"] = ma5

    today_low = SafeFloat(anchors.get("today_low"))
    low_3 = SafeFloat(anchors.get("low_3"))
    prev_low = SafeFloat(anchors.get("prev_day_low"))
    low_5 = SafeFloat(anchors.get("low_5"))

    key_candidates = [v for v in (today_low, low_3) if v > 0]
    anchors["key_support"] = min(key_candidates) if key_candidates else low_5

    next_candidates = [
        v for v in (prev_low, low_5, SafeFloat(indicators.get("boll_lower")))
        if 0 < v < anchors.get("key_support", 9999)
    ]
    anchors["next_support_if_break"] = min(next_candidates) if next_candidates else 0.0

    if current_price > 0:
        pct_floor = round(current_price * 0.97, 2)
        ma20_floor = round(ma20, 2) if ma20 > 0 else pct_floor
        anchors["forbidden_add_above"] = min(pct_floor, ma20_floor)
        if anchors["forbidden_add_above"] >= current_price:
            anchors["forbidden_add_above"] = round(current_price * 0.98, 2)

    today_high = SafeFloat(anchors.get("today_high"))
    high_5 = SafeFloat(anchors.get("high_5"))
    anchors["pressure_t1"] = today_high if today_high > 0 else round(current_price * 1.02, 2)
    anchors["pressure_t2"] = high_5 if high_5 > anchors["pressure_t1"] else round(
        anchors["pressure_t1"] + 0.5, 2,
    )

    return anchors


def BuildPriceContext(
    kline: pd.DataFrame | None,
    indicators: dict[str, Any],
    current_price: float,
    realtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """基于 K 线与指标构建历史价位上下文。"""
    ctx: dict[str, Any] = {"current_price": current_price}
    ctx.update(ExtractRecentPriceLevels(kline, realtime))

    if kline is not None and not kline.empty:
        low_col = "最低" if "最低" in kline.columns else "low"
        high_col = "最高" if "最高" in kline.columns else "high"
        close_col = "收盘" if "收盘" in kline.columns else "close"
        low_s = kline[low_col].astype(float)
        high_s = kline[high_col].astype(float)
        close = kline[close_col].astype(float)
        ctx["high_5"] = float(high_s.tail(5).max())
        ctx["low_5"] = float(low_s.tail(5).min())
        ctx["avg_5"] = float(close.tail(5).mean())
        ctx["high_20"] = float(high_s.tail(20).max())
        ctx["low_20"] = float(low_s.tail(20).min())
        ctx["avg_20"] = float(close.tail(20).mean())
        ctx["high_60"] = float(high_s.max())
        ctx["low_60"] = float(low_s.min())
        ctx["avg_60"] = float(close.mean())
    else:
        for key in (
            "high_5", "low_5", "avg_5",
            "high_20", "low_20", "avg_20",
            "high_60", "low_60", "avg_60",
        ):
            ctx[key] = 0.0

    ctx["ma5"] = SafeFloat(indicators.get("ma5"))
    ctx["ma10"] = SafeFloat(indicators.get("ma10"))
    ctx["ma20"] = SafeFloat(indicators.get("ma20"))
    ctx["ma60"] = SafeFloat(indicators.get("ma60"))
    ctx["boll_upper"] = SafeFloat(indicators.get("boll_upper"))
    ctx["boll_mid"] = SafeFloat(indicators.get("boll_mid"))
    ctx["boll_lower"] = SafeFloat(indicators.get("boll_lower"))

    anchors = CalcPriceAnchors(kline, realtime, indicators, current_price)
    ctx.update(anchors)

    portfolio = CalcPortfolioSummary(current_price)
    if portfolio is not None:
        ctx["cost_lines"] = [lot.cost for lot in portfolio.lots]
        ctx["weighted_cost"] = portfolio.weighted_cost
        ctx["breakeven_price"] = portfolio.breakeven_price
    else:
        ctx["cost_lines"] = []
        ctx["weighted_cost"] = 0.0
        ctx["breakeven_price"] = 0.0

    return ctx


def FormatPriceContextText(
    ctx: dict[str, Any],
    include_cost_lines: bool = True,
) -> list[str]:
    """格式化历史价位文本。"""
    price = ctx.get("current_price", 0)
    lines = [
        f"- 现价：{price:.2f}",
    ]
    if ctx.get("today_open") or ctx.get("today_low") or ctx.get("today_high"):
        lines.append(
            f"- 今日：开 {ctx.get('today_open', 0):.2f} / 高 {ctx.get('today_high', 0):.2f} "
            f"/ 低 {ctx.get('today_low', 0):.2f}"
        )
    if ctx.get("prev_day_low") or ctx.get("prev_day_high"):
        lines.append(
            f"- 昨日：低 {ctx.get('prev_day_low', 0):.2f} / 高 {ctx.get('prev_day_high', 0):.2f}"
        )
    lines.extend([
        f"- 近5日：高 {ctx.get('high_5', 0):.2f} / 低 {ctx.get('low_5', 0):.2f} / 均 {ctx.get('avg_5', 0):.2f}",
        f"- 近20日：高 {ctx.get('high_20', 0):.2f} / 低 {ctx.get('low_20', 0):.2f} / 均 {ctx.get('avg_20', 0):.2f}",
        f"- 近60日：高 {ctx.get('high_60', 0):.2f} / 低 {ctx.get('low_60', 0):.2f} / 均 {ctx.get('avg_60', 0):.2f}",
        f"- 均线：MA5={ctx.get('ma5', 0):.2f} MA10={ctx.get('ma10', 0):.2f} "
        f"MA20={ctx.get('ma20', 0):.2f} MA60={ctx.get('ma60', 0):.2f}",
        f"- 布林带：上 {ctx.get('boll_upper', 0):.2f} / 中 {ctx.get('boll_mid', 0):.2f} / 下 {ctx.get('boll_lower', 0):.2f}",
    ])
    if ctx.get("box_mid"):
        lines.append(
            f"- 箱体（近10日）：低 {ctx.get('box_low', 0):.2f} / 中 {ctx.get('box_mid', 0):.2f} "
            f"/ 高 {ctx.get('box_high', 0):.2f}"
        )
    if ctx.get("prev_day_vwap"):
        lines.append(f"- 昨日VWAP（资金成本线）：{ctx.get('prev_day_vwap', 0):.2f}")
    if ctx.get("key_support"):
        lines.append(
            f"- 关键支撑：{ctx.get('key_support', 0):.2f}  "
            f"跌破后下一档：{ctx.get('next_support_if_break', 0):.2f}"
        )
    if ctx.get("forbidden_add_above"):
        lines.append(f"- 禁加线（此价以上不宜追高加仓）：{ctx.get('forbidden_add_above', 0):.2f}")
    if ctx.get("pressure_t1"):
        lines.append(
            f"- 做T压力：R1={ctx.get('pressure_t1', 0):.2f}  R2={ctx.get('pressure_t2', 0):.2f}"
        )
    cost_lines = ctx.get("cost_lines", [])
    if include_cost_lines and cost_lines:
        costs_str = " / ".join(f"{c:.3f}" for c in cost_lines)
        lines.append(f"- 持仓成本线：{costs_str} 元")
        lines.append(f"- 加权成本/回本价：{ctx.get('weighted_cost', 0):.2f} 元")
    return lines
