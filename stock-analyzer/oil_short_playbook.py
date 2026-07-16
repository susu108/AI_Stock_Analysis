"""石油板块短线专属玩法手册 — 仅 oil_short 组启用。"""

from __future__ import annotations

import config

OIL_SHORT_RISK_HEADER = (
    "以下仅为短线实操思路，不构成投资建议；石油板块地缘消息波动极大，"
    "短线核心：轻仓、快进快出、不格局、严格止损"
)

# 弹性小票 / 中军仓位上限（占账户总资金 %）
_ELASTIC_CODES = frozenset({"600777", "603619"})
_CORE_CODES = frozenset({"600938", "601857"})
SECTOR_MAX_PCT = 20.0
ELASTIC_MAX_PCT = 5.0
CORE_MAX_PCT = 8.0

OIL_SHORT_PLAYBOOK = """
一、总原则（石油板块短线专属）
1. 纯事件脉冲行情：涨靠中东/地缘利好，缓和则集体跳水，绝不长期重仓躺股。
2. 仓位红线：油气赛道合计≤账户总资金20%；弹性小票（新潮600777、中曼603619）单只≤5%；
   中军（中海油600938、中石油601857）单只≤8%。
3. 持仓周期：日内T（当日买当日卖不隔夜）；波段短线1-3天，地缘降温无条件清仓。
4. 不追高：个股单日涨幅>4%放弃新开仓，冲高只做反T减仓，只等回调低吸。

二、入场低吸（正T，先买后卖）
① 日内分时低吸：开盘低开2%-4%且无利空、分时缩量不创新低、黄线支撑；或板块回调1.5%-3%承接有力；
   或回踩5/10日线企稳、RSI<40。禁止：放量连跌、分时无承接、油价同步大跌时抄底。
② 隔夜波段低吸：仅14:40-14:50，布伦特高位且无美伊缓和，回踩日线支撑且收盘站稳20日线，小仓进场。
分批：支撑上沿买50%机动仓；下沿止跌再补50%；禁止第三笔、越跌不加仓。

三、止盈
日内T：反弹3%-5%先卖一半；触及压力/放量滞涨/顶背离清全部T仓；涨停潮可留极小部分。
波段1-3天：浮盈8%卖1/2；15%再卖1/3；余仓最高价回撤7%清仓；出现谈判/通航恢复无论盈亏减仓。

四、止损（绝不扛单）
日内T：跌破支撑下沿0.5元立刻清当天仓；波段：入场价跌5%无条件离场；
黑天鹅：隔夜原油大跌>3%，次日开盘直接减仓。

五、做T
反T（冲高优先）：高开冲高3%+无量，开盘30分钟内卖底仓1/3~1/2，回落2.5%-4%缩量接回。
正T：集体回调缩量止跌后低吸机动仓，日内反弹到位全部卖出不留过夜。

六、时间窗口
9:30-10:00 判强弱优先反T；10:30-11:30 企稳再小仓正T；14:00-14:50 日内T全部平仓，
隔夜仅尾盘少量；14:50后不新开日内仓。

七、避雷
只做上游油气/油服，回避纯炼化；不长持弹性小票；振幅<2.5%不做T；
大盘大跌+油价跳水+地缘缓和三共振→空仓；不重仓单只弹性小盘。

极简：回调缩量支撑分批低吸，冲高分批兑现；日内当日清，波段5%硬止损；
利好冲高反T降本，局势缓和全线离场，不长期格局。
""".strip()


def IsOilShortGroup() -> bool:
    """当前 profile 是否为石油短线监控组。"""
    return str(getattr(config, "STOCK_GROUP", "") or "").strip() == "oil_short"


def IsNonferrousGroup() -> bool:
    """当前 profile 是否为有色周期监控组。"""
    return str(getattr(config, "STOCK_GROUP", "") or "").strip() == "nonferrous"


def MaxPositionPctForStock(stock_code: str | None = None) -> float:
    """返回单只建议仓位上限（占账户资金 %）。"""
    code = (stock_code or config.STOCK_CODE or "").strip()
    if code in _ELASTIC_CODES:
        return ELASTIC_MAX_PCT
    if code in _CORE_CODES:
        return CORE_MAX_PCT
    return ELASTIC_MAX_PCT


def FormatPositionCapNote(stock_code: str | None = None) -> str:
    """仓位上限说明文案。"""
    code = (stock_code or config.STOCK_CODE or "").strip()
    single = MaxPositionPctForStock(code)
    tag = "弹性小票" if code in _ELASTIC_CODES else (
        "中军" if code in _CORE_CODES else "短线标的"
    )
    return (
        f"本股属{tag}，建议仓位≤账户资金{single:.0f}%；"
        f"油气赛道合计≤{SECTOR_MAX_PCT:.0f}%"
    )
