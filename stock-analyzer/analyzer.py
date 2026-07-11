"""分析模块 — 技术面 + 资金面 + 板块面三维度评分。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import config
from utils import SafeFloat, SetupLogger

logger = SetupLogger(config.LOG_LEVEL)


# ---------------------------------------------------------------------------
# 技术指标计算
# ---------------------------------------------------------------------------


def CalcMa(series: pd.Series, period: int) -> pd.Series:
    """计算简单移动平均线。"""
    return series.rolling(window=period, min_periods=period).mean()


def CalcEma(series: pd.Series, period: int) -> pd.Series:
    """计算指数移动平均线。"""
    return series.ewm(span=period, adjust=False).mean()


def CalcMacd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """计算 MACD：DIF, DEA, MACD柱。"""
    ema_fast = CalcEma(close, fast)
    ema_slow = CalcEma(close, slow)
    dif = ema_fast - ema_slow
    dea = CalcEma(dif, signal)
    bar = (dif - dea) * 2
    return dif, dea, bar


def CalcRsi(close: pd.Series, period: int = 6) -> pd.Series:
    """计算 RSI。"""
    delta = close.diff()
    gains = delta.where(delta > 0, 0.0)
    losses = (-delta).where(delta < 0, 0.0)
    avg_gain = gains.rolling(window=period, min_periods=period).mean()
    avg_loss = losses.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def CalcKdj(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    n: int = 9,
    m1: int = 3,
    m2: int = 3,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """计算 KDJ。"""
    llv = low.rolling(window=n, min_periods=n).min()
    hhv = high.rolling(window=n, min_periods=n).max()
    rsv = ((close - llv) / (hhv - llv).replace(0, np.nan)) * 100
    rsv = rsv.fillna(50.0)
    k = rsv.ewm(com=m1 - 1, adjust=False).mean()
    d = k.ewm(com=m2 - 1, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


def CalcBoll(
    close: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """计算布林带：上轨、中轨、下轨。"""
    mid = CalcMa(close, period)
    std = close.rolling(window=period, min_periods=period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return upper, mid, lower


def CalcVolumeRatio(volume: pd.Series, period: int = 10) -> float:
    """计算量比 = 今日成交量 / 近 period 日均量。"""
    if len(volume) < period + 1:
        return 1.0
    avg_vol = volume.iloc[-(period + 1) : -1].mean()
    if avg_vol == 0:
        return 1.0
    return float(volume.iloc[-1] / avg_vol)


def ComputeIndicators(kline: pd.DataFrame) -> dict[str, Any]:
    """基于 K 线计算全部技术指标。"""
    close = kline["收盘"].astype(float)
    high = kline["最高"].astype(float)
    low = kline["最低"].astype(float)
    volume = kline["成交量"].astype(float)

    ma5 = CalcMa(close, 5)
    ma10 = CalcMa(close, 10)
    ma20 = CalcMa(close, 20)
    ma60 = CalcMa(close, 60)
    dif, dea, macd_bar = CalcMacd(close)
    rsi = CalcRsi(close, 6)
    k, d, j = CalcKdj(high, low, close)
    boll_upper, boll_mid, boll_lower = CalcBoll(close)
    vol_ratio = CalcVolumeRatio(volume)

    price = float(close.iloc[-1])
    prev_close = float(close.iloc[-2]) if len(close) >= 2 else price
    is_up = price >= prev_close

    boll_range = float(boll_upper.iloc[-1] - boll_lower.iloc[-1])
    boll_pos = (
        (price - float(boll_lower.iloc[-1])) / boll_range if boll_range > 0 else 0.5
    )

    return {
        "price": price,
        "prev_close": prev_close,
        "is_up": is_up,
        "ma5": float(ma5.iloc[-1]) if not np.isnan(ma5.iloc[-1]) else price,
        "ma10": float(ma10.iloc[-1]) if not np.isnan(ma10.iloc[-1]) else price,
        "ma20": float(ma20.iloc[-1]) if not np.isnan(ma20.iloc[-1]) else price,
        "ma60": float(ma60.iloc[-1]) if not np.isnan(ma60.iloc[-1]) else price,
        "ma5_prev": float(ma5.iloc[-2]) if len(ma5) >= 2 and not np.isnan(ma5.iloc[-2]) else price,
        "ma10_prev": float(ma10.iloc[-2]) if len(ma10) >= 2 and not np.isnan(ma10.iloc[-2]) else price,
        "dif": float(dif.iloc[-1]) if not np.isnan(dif.iloc[-1]) else 0.0,
        "dea": float(dea.iloc[-1]) if not np.isnan(dea.iloc[-1]) else 0.0,
        "dif_prev": float(dif.iloc[-2]) if len(dif) >= 2 and not np.isnan(dif.iloc[-2]) else 0.0,
        "dea_prev": float(dea.iloc[-2]) if len(dea) >= 2 and not np.isnan(dea.iloc[-2]) else 0.0,
        "macd_bar": float(macd_bar.iloc[-1]) if not np.isnan(macd_bar.iloc[-1]) else 0.0,
        "macd_bar_prev": float(macd_bar.iloc[-2]) if len(macd_bar) >= 2 and not np.isnan(macd_bar.iloc[-2]) else 0.0,
        "rsi": float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0,
        "k": float(k.iloc[-1]) if not np.isnan(k.iloc[-1]) else 50.0,
        "d": float(d.iloc[-1]) if not np.isnan(d.iloc[-1]) else 50.0,
        "j": float(j.iloc[-1]) if not np.isnan(j.iloc[-1]) else 50.0,
        "k_prev": float(k.iloc[-2]) if len(k) >= 2 and not np.isnan(k.iloc[-2]) else 50.0,
        "d_prev": float(d.iloc[-2]) if len(d) >= 2 and not np.isnan(d.iloc[-2]) else 50.0,
        "boll_upper": float(boll_upper.iloc[-1]) if not np.isnan(boll_upper.iloc[-1]) else price,
        "boll_mid": float(boll_mid.iloc[-1]) if not np.isnan(boll_mid.iloc[-1]) else price,
        "boll_lower": float(boll_lower.iloc[-1]) if not np.isnan(boll_lower.iloc[-1]) else price,
        "boll_pos": boll_pos,
        "vol_ratio": vol_ratio,
        "high_20": float(high.tail(20).max()),
        "low_20": float(low.tail(20).min()),
        "kline": kline,
    }


# ---------------------------------------------------------------------------
# 三维度评分
# ---------------------------------------------------------------------------


def AnalyzeTechnical(ind: dict[str, Any]) -> tuple[float, list[str]]:
    """技术面分析，返回 (score, signals)。"""
    score = 0.0
    signals: list[str] = []
    price = ind["price"]
    ma5, ma10, ma20 = ind["ma5"], ind["ma10"], ind["ma20"]
    ma5_prev, ma10_prev = ind["ma5_prev"], ind["ma10_prev"]

    if ma5 > ma10 > ma20:
        score += 20
        signals.append(f"均线多头排列（MA5={ma5:.2f} > MA10={ma10:.2f} > MA20={ma20:.2f}）")
    elif ma5 < ma10 < ma20:
        score -= 20
        signals.append(f"均线空头排列（MA5={ma5:.2f} < MA10={ma10:.2f} < MA20={ma20:.2f}）")

    if price > ma20:
        score += 10
        signals.append(f"股价({price:.2f})站上MA20({ma20:.2f})")
    elif price < ma20:
        score -= 10
        signals.append(f"股价({price:.2f})跌破MA20({ma20:.2f})")

    if ma5_prev <= ma10_prev and ma5 > ma10:
        score += 15
        signals.append("MA5金叉MA10")
    elif ma5_prev >= ma10_prev and ma5 < ma10:
        score -= 15
        signals.append("MA5死叉MA10")

    dif, dea = ind["dif"], ind["dea"]
    dif_prev, dea_prev = ind["dif_prev"], ind["dea_prev"]
    bar, bar_prev = ind["macd_bar"], ind["macd_bar_prev"]

    if dif_prev <= dea_prev and dif > dea:
        score += 20
        signals.append("MACD金叉")
    elif dif_prev >= dea_prev and dif < dea:
        score -= 20
        signals.append("MACD死叉")

    if bar > bar_prev and bar > 0:
        score += 10
        signals.append("MACD红柱放大")
    elif bar < bar_prev and bar < 0:
        score -= 10
        signals.append("MACD绿柱放大")
    elif bar > 0:
        score += 5
        signals.append("MACD红柱")
    elif bar < 0:
        score -= 5
        signals.append("MACD绿柱")

    rsi_val = ind["rsi"]
    if rsi_val > 80:
        score -= 15
        signals.append(f"RSI(6)={rsi_val:.1f} 超买")
    elif rsi_val < 20:
        score += 15
        signals.append(f"RSI(6)={rsi_val:.1f} 超卖")
    elif rsi_val > 60:
        score += 5
        signals.append(f"RSI(6)={rsi_val:.1f} 偏强")
    elif rsi_val < 40:
        score -= 5
        signals.append(f"RSI(6)={rsi_val:.1f} 偏弱")

    k_val, d_val, j_val = ind["k"], ind["d"], ind["j"]
    k_prev, d_prev = ind["k_prev"], ind["d_prev"]
    if k_prev <= d_prev and k_val > d_val:
        score += 15
        signals.append("KDJ金叉")
    elif k_prev >= d_prev and k_val < d_val:
        score -= 15
        signals.append("KDJ死叉")

    if j_val > 100:
        score -= 10
        signals.append(f"KDJ-J={j_val:.1f} 超买")
    elif j_val < 0:
        score += 10
        signals.append(f"KDJ-J={j_val:.1f} 超卖")

    boll_pos = ind["boll_pos"]
    boll_upper = ind["boll_upper"]
    boll_lower = ind["boll_lower"]
    if boll_pos > 0.85:
        score -= 10
        signals.append(f"触及布林上轨({boll_upper:.2f})，短期有压力")
    elif boll_pos < 0.15:
        score += 10
        signals.append(f"触及布林下轨({boll_lower:.2f})，短期有支撑")

    vol_ratio = ind["vol_ratio"]
    is_up = ind["is_up"]
    if vol_ratio > 1.5 and is_up:
        score += 15
        signals.append(f"放量上涨（量比={vol_ratio:.2f}）")
    elif vol_ratio > 1.5 and not is_up:
        score -= 15
        signals.append(f"放量下跌（量比={vol_ratio:.2f}）")
    elif vol_ratio < 0.5 and is_up:
        score += 5
        signals.append(f"缩量上涨（量比={vol_ratio:.2f}）")
    elif vol_ratio < 0.5 and not is_up:
        score -= 5
        signals.append(f"缩量下跌（量比={vol_ratio:.2f}）")

    score = max(-100.0, min(100.0, score))
    return score, signals


def AnalyzeFundFlow(fund_flow: pd.DataFrame | None) -> tuple[float, list[str]]:
    """资金面分析，返回 (score, signals)。"""
    if fund_flow is None or fund_flow.empty:
        return 0.0, ["资金面数据不足"]

    score = 0.0
    signals: list[str] = []

    main_col = "主力净流入-净额"
    super_col = "超大单净流入-净额"
    if main_col not in fund_flow.columns:
        return 0.0, ["资金面数据格式异常"]

    recent = fund_flow.tail(5)
    main_flows = [SafeFloat(v) for v in recent[main_col]]
    positive_days = sum(1 for v in main_flows if v > 0)
    negative_days = sum(1 for v in main_flows if v < 0)
    total_flow = sum(main_flows)
    latest_flow = main_flows[-1] if main_flows else 0.0

    if all(abs(v) < 1 for v in main_flows):
        return 0.0, ["资金面数据不足（未能获取有效流向数据）"]

    if positive_days >= 4:
        score += 30
        signals.append(f"近5日主力净流入 {positive_days} 天，资金持续流入")
    elif negative_days >= 4:
        score -= 30
        signals.append(f"近5日主力净流出 {negative_days} 天，资金持续流出")
    elif positive_days > negative_days:
        score += 10
        signals.append(f"近5日主力净流入 {positive_days} 天多于流出 {negative_days} 天")
    elif negative_days > positive_days:
        score -= 10
        signals.append(f"近5日主力净流出 {negative_days} 天多于流入 {positive_days} 天")

    if latest_flow > 0:
        score += 15
        signals.append(f"最近1日主力净流入 {latest_flow / 1e4:.0f} 万元")
    elif latest_flow < 0:
        score -= 15
        signals.append(f"最近1日主力净流出 {abs(latest_flow) / 1e4:.0f} 万元")

    if super_col in fund_flow.columns:
        super_flows = [SafeFloat(v) for v in recent[super_col]]
        super_positive = sum(1 for v in super_flows if v > 0)
        if all(abs(v) < 1 for v in super_flows):
            pass
        elif super_positive >= 4:
            score += 20
            signals.append(f"超大单（机构）近5日 {super_positive} 天净买入")
        elif super_positive <= 1 and any(v < 0 for v in super_flows):
            score -= 20
            signals.append(f"超大单（机构）近5日仅 {super_positive} 天净买入")

    if total_flow > 1e7:
        score += 10
        signals.append(f"近5日累计主力净流入 {total_flow / 1e4:.0f} 万元")
    elif total_flow < -1e7:
        score -= 10
        signals.append(f"近5日累计主力净流出 {abs(total_flow) / 1e4:.0f} 万元")

    score = max(-100.0, min(100.0, score))
    return score, signals


def AnalyzeSector(
    concept: pd.DataFrame | None,
    industry: pd.DataFrame | None,
) -> tuple[float, list[str]]:
    """板块面分析，返回 (score, signals)。"""
    score = 0.0
    signals: list[str] = []

    if concept is not None and not concept.empty and "涨跌幅" in concept.columns:
        changes = concept["涨跌幅"].astype(float)
        up_count = int((changes > 0).sum())
        down_count = int((changes < 0).sum())
        total = up_count + down_count
        up_ratio = up_count / total if total > 0 else 0.5
        signals.append(f"概念板块：{up_count}涨 / {down_count}跌")

        if up_ratio > 0.7:
            score += 25
            signals.append("市场普涨，概念板块上涨占比超70%")
        elif up_ratio < 0.3:
            score -= 25
            signals.append("市场普跌，概念板块上涨占比不足30%")

        top_change = float(changes.iloc[0]) if len(changes) > 0 else 0.0
        bottom_change = float(changes.iloc[-1]) if len(changes) > 0 else 0.0
        if top_change > 3:
            score += 15
            top_name = str(concept.iloc[0].get("板块名称", ""))
            signals.append(f"最强概念「{top_name}」涨幅{top_change:.2f}%")
        if bottom_change < -3:
            score -= 15
            bottom_name = str(concept.iloc[-1].get("板块名称", ""))
            signals.append(f"最弱概念「{bottom_name}」跌幅{bottom_change:.2f}%")
    else:
        signals.append("概念板块数据不足")

    if industry is not None and not industry.empty and "涨跌幅" in industry.columns:
        ind_changes = industry["涨跌幅"].astype(float)
        top_ind = float(ind_changes.iloc[0]) if len(ind_changes) > 0 else 0.0
        bottom_ind = float(ind_changes.iloc[-1]) if len(ind_changes) > 0 else 0.0
        if top_ind > 2:
            score += 10
            top_name = str(industry.iloc[0].get("板块名称", ""))
            signals.append(f"领涨行业「{top_name}」涨幅{top_ind:.2f}%")
        if bottom_ind < -2:
            score -= 10
            bottom_name = str(industry.iloc[-1].get("板块名称", ""))
            signals.append(f"领跌行业「{bottom_name}」跌幅{bottom_ind:.2f}%")

    score = max(-100.0, min(100.0, score))
    return score, signals


def CalcConfidence(abs_score: float) -> int:
    """根据绝对分数计算信心指数 1~5。"""
    if abs_score >= 50:
        return 5
    if abs_score >= 35:
        return 4
    if abs_score >= 20:
        return 3
    if abs_score >= 10:
        return 2
    return 1


def CalcDirection(weighted: float) -> tuple[str, str]:
    """根据加权分数判定方向和图标。"""
    if weighted > 25:
        return "看涨", "⬆️"
    if weighted < -25:
        return "看跌", "⬇️"
    if weighted > 0:
        return "震荡偏涨", "↗️"
    if weighted < 0:
        return "震荡偏跌", "↘️"
    return "震荡", "➡️"


WEIGHT_TECH = 0.35
WEIGHT_FUND = 0.30
WEIGHT_SECTOR = 0.20
WEIGHT_NEWS = 0.15


def MergeNewsIntoAnalysis(
    analysis: dict[str, Any],
    news_bundle: dict[str, Any] | None,
) -> dict[str, Any]:
    """将资讯面评分融入四维度加权，重算方向/区间/信心。"""
    tech_score = float(analysis.get("tech_score", 0.0))
    fund_score = float(analysis.get("fund_score", 0.0))
    sector_score = float(analysis.get("sector_score", 0.0))
    bundle = news_bundle or {}
    news_score = float(bundle.get("news_score", 0.0))
    news_signals = list(bundle.get("news_signals", []))

    base_weighted = (
        tech_score * 0.40
        + fund_score * 0.35
        + sector_score * 0.25
    )
    weighted = (
        tech_score * WEIGHT_TECH
        + fund_score * WEIGHT_FUND
        + sector_score * WEIGHT_SECTOR
        + news_score * WEIGHT_NEWS
    )
    direction, direction_icon = CalcDirection(weighted)
    est_change = weighted / 100 * 5
    change_low = est_change - 1.5
    change_high = est_change + 1.5
    confidence = CalcConfidence(abs(weighted))

    merged = {
        **analysis,
        "news_score": news_score,
        "news_signals": news_signals,
        "base_weighted_score": base_weighted,
        "weighted_score": weighted,
        "direction": direction,
        "direction_icon": direction_icon,
        "change_low": change_low,
        "change_high": change_high,
        "confidence": confidence,
    }

    logger.info(
        "四维度融合 — 综合:%.1f 技术:%.1f 资金:%.1f 板块:%.1f 资讯:%.1f → %s",
        weighted,
        tech_score,
        fund_score,
        sector_score,
        news_score,
        direction,
    )
    return merged


def AnalyzeAll(data: dict[str, Any]) -> dict[str, Any]:
    """执行三维度综合分析，返回完整分析结果。

    日常报告需在资讯分析后调用 MergeNewsIntoAnalysis 融合资讯面（15%权重）。
    """
    kline = data.get("kline")
    indicators: dict[str, Any] = {}

    if kline is not None and not kline.empty:
        indicators = ComputeIndicators(kline)
        tech_score, tech_signals = AnalyzeTechnical(indicators)
    else:
        tech_score, tech_signals = 0.0, ["K线数据不足，技术面无法分析"]

    fund_score, fund_signals = AnalyzeFundFlow(data.get("fund_flow"))
    sector_score, sector_signals = AnalyzeSector(
        data.get("concept"), data.get("industry")
    )

    weighted = tech_score * 0.40 + fund_score * 0.35 + sector_score * 0.25
    direction, direction_icon = CalcDirection(weighted)
    est_change = weighted / 100 * 5
    change_low = est_change - 1.5
    change_high = est_change + 1.5
    confidence = CalcConfidence(abs(weighted))

    logger.info(
        "分析完成 — 综合:%.1f 技术:%.1f 资金:%.1f 板块:%.1f → %s",
        weighted,
        tech_score,
        fund_score,
        sector_score,
        direction,
    )

    return {
        "tech_score": tech_score,
        "fund_score": fund_score,
        "sector_score": sector_score,
        "weighted_score": weighted,
        "direction": direction,
        "direction_icon": direction_icon,
        "change_low": change_low,
        "change_high": change_high,
        "confidence": confidence,
        "tech_signals": tech_signals,
        "fund_signals": fund_signals,
        "sector_signals": sector_signals,
        "indicators": indicators,
    }
