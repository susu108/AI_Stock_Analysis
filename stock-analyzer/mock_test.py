"""模拟数据测试 — 无需网络，验证报告格式。"""

from __future__ import annotations

import os

import config
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from advisor import GenerateAdvice
from analyzer import AnalyzeAll, ComputeIndicators, MergeNewsIntoAnalysis
from data_fetcher import MergeRealtimeIntoKline
from dingtalk_pusher import (
    BuildDingTalkReportMarkdown,
    BuildReportMarkdown,
    ResolveTradeDisplayLevels,
)
from market_context import ResolvePredictionHorizon, ResolveSessionAndHorizon
from move_context import BuildMoveContext
from news_analyzer import AnalyzeNewsImpact, CalcNewsScore, GetBackgroundNewsItems
from price_move_analyzer import (
    BuildMoveContextPrompt,
    BuildRulePriceMoveFallback,
)
from scheduler_guard import BuildPushSlotKey
from stock_profile import ApplyStockProfile, FilterProfilesByCode, FilterProfilesByGroup


def BuildMockKline(days: int = 60) -> pd.DataFrame:
    """构造模拟 K 线数据（与方案 §11 示例趋势一致）。"""
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days, freq="B")
    base_price = 63.0
    prices: list[float] = []
    for i in range(days):
        trend = 0.002 if i > days // 3 else 0.001
        noise = np.random.normal(0, 0.8)
        base_price = base_price * (1 + trend) + noise
        base_price = max(base_price, 55.0)
        prices.append(base_price)

    closes = np.array(prices)
    opens = closes * (1 + np.random.uniform(-0.01, 0.01, days))
    highs = np.maximum(opens, closes) * (1 + np.random.uniform(0, 0.02, days))
    lows = np.minimum(opens, closes) * (1 - np.random.uniform(0, 0.02, days))
    volumes = np.random.randint(100000, 500000, days).astype(float)
    volumes[-1] = int(volumes[-5:].mean() * 1.85)

    return pd.DataFrame({
        "日期": dates.strftime("%Y-%m-%d"),
        "开盘": np.round(opens, 2),
        "收盘": np.round(closes, 2),
        "最高": np.round(highs, 2),
        "最低": np.round(lows, 2),
        "成交量": volumes,
        "成交额": volumes * closes,
        "振幅": np.round(np.random.uniform(1, 5, days), 2),
        "涨跌幅": np.round(np.diff(closes, prepend=closes[0]) / closes * 100, 2),
        "涨跌额": np.round(np.diff(closes, prepend=closes[0]), 2),
        "换手率": np.round(np.random.uniform(2, 8, days), 2),
    })


def BuildMockFundFlow() -> pd.DataFrame:
    """构造模拟资金流向数据。"""
    dates = pd.date_range(end=pd.Timestamp.today(), periods=100, freq="B")
    recent_main = [2410000, 520000, 380000, 310000, 290000]
    recent_super = [1800000, 420000, 250000, 200000, 150000]
    main_all = [np.random.randint(-500000, 800000) for _ in range(95)] + recent_main
    super_all = [np.random.randint(-300000, 500000) for _ in range(95)] + recent_super

    return pd.DataFrame({
        "日期": dates.strftime("%Y-%m-%d"),
        "主力净流入-净额": main_all,
        "超大单净流入-净额": super_all,
        "大单净流入-净额": [m - s for m, s in zip(main_all, super_all)],
        "中单净流入-净额": [np.random.randint(-200000, 200000) for _ in range(100)],
        "小单净流入-净额": [np.random.randint(-100000, 100000) for _ in range(100)],
    })


def BuildMockConceptBoards() -> pd.DataFrame:
    """构造模拟概念板块数据。"""
    names = [
        "医疗信息化", "家用医疗", "中药", "疫苗", "口腔医疗",
        "AI概念", "新能源", "半导体", "军工", "消费",
        "房地产", "银行", "保险", "钢铁", "煤炭",
        "石油", "化工", "建材", "农业", "传媒",
    ]
    changes = [3.89, 3.83, 3.26, 2.70, 2.32, 1.85, 1.20, 0.95, 0.60, 0.30,
               -0.20, -0.50, -0.80, -1.10, -1.50, -1.80, -2.10, -2.50, -2.80, -3.20]
    return pd.DataFrame({
        "排名": range(1, len(names) + 1),
        "板块名称": names,
        "板块代码": [f"BK{i:04d}" for i in range(len(names))],
        "最新价": np.random.uniform(1000, 5000, len(names)).round(2),
        "涨跌幅": changes,
        "涨跌额": [c * 10 for c in changes],
        "成交量": np.random.randint(1000000, 5000000, len(names)),
        "换手率": np.random.uniform(1, 5, len(names)).round(2),
    })


def BuildMockIndustryBoards() -> pd.DataFrame:
    """构造模拟行业板块数据。"""
    names = ["医药生物", "电子", "计算机", "机械设备", "化工",
             "银行", "房地产", "钢铁", "煤炭", "石油石化"]
    changes = [2.50, 2.10, 1.80, 1.20, 0.80, -0.30, -0.80, -1.50, -2.10, -2.80]
    return pd.DataFrame({
        "排名": range(1, len(names) + 1),
        "板块名称": names,
        "板块代码": [f"BK{i:04d}" for i in range(len(names))],
        "最新价": np.random.uniform(1000, 5000, len(names)).round(2),
        "涨跌幅": changes,
    })


def BuildMockRealtime() -> dict:
    """构造模拟实时行情。"""
    return {
        "code": "301075",
        "name": "多瑞生物",
        "price": 61.67,
        "change_pct": 5.20,
        "change_amt": 3.05,
        "open": 62.80,
        "high": 64.10,
        "low": 62.50,
        "prev_close": 62.62,
        "volume": 2850000,
        "amount": 180000000,
        "turnover": 4.32,
        "volume_ratio": 1.85,
        "pe": 35.6,
        "pb": 3.28,
        "amplitude": 2.56,
        "total_mv": 9.15e9,
        "circ_mv": 6.8e9,
    }


def BuildMockNews() -> dict[str, Any]:
    """构造模拟多源资讯。"""
    stock = [
        {
            "title": "多瑞生物：子公司获得药品注册证书",
            "content": "公司公告子公司某产品获得药品注册证书，有望丰富产品线。",
            "time": "2026-07-10 08:30",
            "source": "东方财富",
            "category": "stock",
        },
        {
            "title": "多瑞医药向原控股股东借款2.8亿元补充流动资金",
            "content": "公司借款用于补充流动资金，按LPR计息。",
            "time": "2026-07-06 17:27",
            "source": "财中社",
            "category": "stock",
        },
    ]
    sector = [
        {
            "title": "医药板块震荡走强，创新药概念受资金关注",
            "content": "今日医药板块整体活跃，创新药、医疗器械涨幅居前。",
            "time": "2026-07-10 09:15",
            "source": "东方财富",
            "category": "sector",
        },
        {
            "title": "家用医疗器械需求持续增长，板块估值修复",
            "content": "家用医疗概念多股上涨，机构看好长期空间。",
            "time": "2026-07-10 10:00",
            "source": "东方财富",
            "category": "sector",
        },
    ]
    policy = [
        {
            "title": "国家基药目录调整：司美格鲁肽注射液（降糖）正式纳入",
            "content": "三部门联合发文，司美格鲁肽注射液降糖适应症纳入国家基药目录，9月起基层医院统一配备。",
            "time": "20260709",
            "source": "央视新闻",
            "category": "policy",
        },
        {
            "title": "国家药监局发布创新药审评审批优化措施",
            "content": "政策鼓励创新药加速上市，利好生物医药企业。",
            "time": "20260710",
            "source": "央视新闻",
            "category": "policy",
        },
        {
            "title": "医保目录调整方案征求意见，创新药纳入比例提升",
            "content": "国家医保局发布目录调整征求意见稿，创新药谈判准入机制进一步优化。",
            "time": "20260709",
            "source": "新华社",
            "category": "policy",
        },
    ]
    web_search = [
        {
            "title": "多瑞生物 GLP-1多肽原料药布局受关注",
            "content": "公司通过子公司布局司美格鲁肽、替尔泊肽等多肽原料药，拥有吨级合成产能。",
            "time": "2026-07-10 09:00",
            "source": "Tavily",
            "category": "web_search",
            "url": "https://example.com/news1",
        },
        {
            "title": "7月10日减肥药概念大涨 板块资金涌入",
            "content": "GLP-1减肥药指数大涨，多只龙头涨停，低位医药小票获资金挖掘。",
            "time": "2026-07-10 10:30",
            "source": "Tavily",
            "category": "web_search",
            "url": "https://example.com/news2",
        },
    ]
    macro = [
        {
            "title": "[中国] 6月CPI同比公布",
            "content": "公布:0.1 预期:0.2 前值:0.0",
            "time": "09:30",
            "source": "经济日历",
            "category": "macro",
        },
    ]
    items = stock + sector + policy + macro + web_search
    return {
        "items": items,
        "stock": stock,
        "sector": sector,
        "policy": policy,
        "macro": macro,
        "web_search": web_search,
        "sector_keywords": ["医药", "医疗", "创新药"],
        "sector_snapshot": [
            "概念板块TOP5：",
            "  - 医疗信息化 +3.89%",
            "  - 家用医疗 +3.83%",
        ],
    }


def RunPredictionHorizonTest(
    data: dict[str, Any],
    analysis: dict[str, Any],
    news_bundle: dict[str, Any],
    session_label: str,
) -> tuple[str, dict[str, Any]]:
    """按指定时段生成报告并返回。"""
    horizon = ResolvePredictionHorizon(session_label)
    advice = GenerateAdvice(
        analysis,
        data,
        news_items=news_bundle["items"],
        news_bundle=news_bundle,
        session_label=session_label,
        mode="daily",
        prediction_horizon=horizon,
    )
    report = BuildReportMarkdown(
        data, analysis, advice, session_label=session_label, report_mode="daily",
    )
    return report, advice


def VerifyLayeredPredictions(report: str, expected_near_keyword: str) -> bool:
    """验证三层预测结构。"""
    near_ok = (
        expected_near_keyword in report
        or "下交易日预测" in report
    )
    return (
        near_ok
        and "短期（1~2周）" in report
        and "长期（1~3月）" in report
        and report.count("**操作建议**") >= 3
    )


def RunHorizonResolverTests(
    data: dict[str, Any],
    analysis: dict[str, Any],
    news_bundle: dict[str, Any],
) -> tuple[bool, bool, bool]:
    """验证非交易日与周五 horizon 解析及钉钉展示。"""
    sat_label, sat_horizon = ResolveSessionAndHorizon(
        "下午盘中", check_date=date(2026, 7, 11),
    )
    sat_ok = (
        sat_label == "周末休市"
        and sat_horizon.get("near_term_label") == "下交易日预测"
        and sat_horizon.get("is_trading_day") is False
        and "7月13日" in str(sat_horizon.get("near_term_target", ""))
        and "周一" in str(sat_horizon.get("near_term_target", ""))
    )

    fri_label, fri_horizon = ResolveSessionAndHorizon(
        "收盘后", check_date=date(2026, 7, 10),
    )
    fri_ok = (
        fri_label == "收盘后"
        and fri_horizon.get("is_trading_day") is True
        and fri_horizon.get("near_term_label") == "下交易日预测"
        and "7月13日" in str(fri_horizon.get("near_term_target", ""))
        and "周六" not in str(fri_horizon.get("near_term_target", ""))
    )

    sat_advice = GenerateAdvice(
        analysis,
        data,
        news_items=news_bundle["items"],
        news_bundle=news_bundle,
        session_label=sat_label,
        mode="daily",
        prediction_horizon=sat_horizon,
    )
    sat_advice["news_summary"] = "休市期间政策面偏暖，关注下一交易日开盘"
    sat_report = BuildDingTalkReportMarkdown(
        data, analysis, sat_advice,
        session_label=sat_label,
        report_mode="daily",
    )
    sat_dingtalk_ok = (
        "周末休市" in sat_report
        and "下交易日预测" in sat_report
        and "预测对象" in sat_report
        and "休市中" in sat_report
        and "资讯研判" in sat_report
        and "休市期间政策面偏暖" in sat_report
    )
    return sat_ok, fri_ok, sat_dingtalk_ok


def RunKlineMergeTest() -> bool:
    """验证盘中实时价融合进 K 线后，技术指标现价与 spot 一致。"""
    kline = BuildMockKline(20)
    kline = kline[kline["日期"] <= "2026-07-10"].copy()
    if kline.empty:
        return False

    realtime: dict[str, Any] = {
        "price": 65.50,
        "prev_close": 61.67,
        "open": 62.00,
        "high": 66.00,
        "low": 61.50,
        "volume": 3000000.0,
        "amount": 190000000.0,
        "turnover": 5.0,
    }
    session_time = datetime(2026, 7, 13, 10, 0)
    merged = MergeRealtimeIntoKline(kline, realtime, check_time=session_time)
    if merged is None or merged.empty:
        return False
    last_row = merged.iloc[-1]
    if str(last_row["日期"])[:10] != "2026-07-13":
        return False
    indicators = ComputeIndicators(merged)
    return abs(float(indicators["price"]) - 65.50) < 0.01


def RunTailHorizonTest() -> bool:
    """验证 14:30 尾盘标签的 horizon 为今日预测。"""
    horizon = ResolvePredictionHorizon("尾盘")
    _, session_horizon = ResolveSessionAndHorizon("尾盘", check_date=date(2026, 7, 13))
    return (
        horizon.get("is_market_open") is True
        and horizon.get("near_term_horizon") == "today"
        and session_horizon.get("is_market_open") is True
        and "今日" in str(session_horizon.get("near_term_target", ""))
    )


def BuildMondayScenarioKline() -> pd.DataFrame:
    """构造周一盘中 K 线：上一交易日为周五 7/10 大涨 8.10%。"""
    return pd.DataFrame({
        "日期": ["2026-07-08", "2026-07-09", "2026-07-10", "2026-07-13"],
        "开盘": [54.0, 55.0, 56.5, 61.0],
        "收盘": [55.0, 56.0, 61.67, 60.15],
        "最高": [55.5, 56.5, 62.0, 61.5],
        "最低": [53.5, 54.5, 56.0, 59.8],
        "成交量": [200000.0, 220000.0, 350000.0, 280000.0],
        "成交额": [1.1e7, 1.2e7, 2.1e7, 1.7e7],
        "涨跌幅": [1.2, 1.8, 8.10, -2.46],
        "涨跌额": [0.65, 0.98, 4.61, -1.52],
        "换手率": [3.0, 3.2, 5.1, 4.0],
    })


def RunTradingDayDateLabelTest() -> bool:
    """验证周一运行时上一交易日标注为周五 7/10，且 prompt 不含裸「昨日涨跌」。"""
    saved_llm = config.LLM_ENABLED
    saved_min_pct = config.PRICE_MOVE_MIN_PCT
    try:
        config.LLM_ENABLED = False
        config.PRICE_MOVE_MIN_PCT = 0.5

        kline = BuildMondayScenarioKline()
        data: dict[str, Any] = {
            "realtime": {
                "price": 60.15,
                "change_pct": -2.46,
                "change_amt": -1.52,
                "prev_close": 61.67,
                "open": 61.0,
                "high": 61.5,
                "low": 59.8,
                "amount": 170000000,
                "volume_ratio": 1.2,
                "circ_mv": 6.8e9,
            },
            "kline": kline,
            "concept": BuildMockConceptBoards(),
            "industry": BuildMockIndustryBoards(),
            "fund_flow": BuildMockFundFlow(),
        }
        news_bundle = AnalyzeNewsImpact(BuildMockNews(), data)
        analysis = AnalyzeAll(data)
        analysis = MergeNewsIntoAnalysis(analysis, news_bundle)

        move_ctx = BuildMoveContext(data, analysis, news_bundle)
        ctx_ok = (
            move_ctx.get("prev_trading_date") == "2026-07-10"
            and "周五" in str(move_ctx.get("prev_trading_date_label", ""))
            and abs(float(move_ctx.get("prev_trading_change_pct", 0) or 0) - 8.10) < 0.01
            and len(move_ctx.get("recent_trading_days") or []) >= 2
        )

        prompt = BuildMoveContextPrompt(move_ctx)
        prompt_ok = (
            "7月10日" in prompt
            and "昨日涨跌" not in prompt
            and "上一交易日" in prompt
            and "2026-07-10" in prompt
        )

        fallback = BuildRulePriceMoveFallback(move_ctx)
        risks_dim = next(
            (d for d in (fallback.get("dimensions") or []) if d.get("id") == "risks"),
            {},
        )
        risk_text = " ".join(
            str(p.get("content", "")) for p in (risks_dim.get("points") or [])
        )
        fallback_ok = "7月10日" in risk_text and "8.10" in risk_text

        news_time_ok = any(
            str(item.get("time", "")).startswith("2026-07-10")
            for item in (move_ctx.get("impactful_news") or [])
            + (move_ctx.get("background_news") or [])
        ) and "2026-07-10" in prompt

        return ctx_ok and prompt_ok and fallback_ok and news_time_ok
    finally:
        config.LLM_ENABLED = saved_llm
        config.PRICE_MOVE_MIN_PCT = saved_min_pct


def RunMultiStockProfileTest() -> bool:
    """验证多股 STOCK_PROFILES 解析、profile 切换与推送槽位隔离。"""
    saved_profiles_env = os.environ.get("STOCK_PROFILES")
    saved_code = config.STOCK_CODE
    saved_name = config.STOCK_NAME
    saved_themes = list(config.STOCK_THEMES)
    saved_positions = list(config.POSITIONS)
    try:
        os.environ["STOCK_PROFILES"] = (
            '[{"code":"301075","name":"多瑞生物","market":"sz","themes":["GLP-1"],'
            '"business":"测试A","positions":[{"account":"A","cost":60,"shares":100}],'
            '"channel":"default","group":"pharma"},'
            '{"code":"301201","name":"诚达药业","market":"sz","themes":["CDMO"],'
            '"business":"测试B","positions":[],'
            '"channel":"default","group":"pharma"}]'
        )
        profiles = config.ResolveStockProfiles()
        parse_ok = (
            len(profiles) == 2
            and profiles[0].get("code") == "301075"
            and profiles[1].get("code") == "301201"
            and profiles[1].get("positions") == []
            and profiles[0].get("group") == "pharma"
        )

        filtered = FilterProfilesByCode(profiles, "301201")
        filter_ok = len(filtered) == 1 and filtered[0].get("name") == "诚达药业"

        with ApplyStockProfile(profiles[1]):
            switch_ok = (
                config.STOCK_CODE == "301201"
                and config.STOCK_NAME == "诚达药业"
                and config.STOCK_THEMES == ["CDMO"]
                and config.POSITIONS == []
            )
        restore_ok = config.STOCK_CODE == saved_code and config.STOCK_NAME == saved_name

        key_a = BuildPushSlotKey("09:00", "开盘前", stock_code="301075")
        key_b = BuildPushSlotKey("09:00", "开盘前", stock_code="301201")
        slot_ok = key_a != key_b and key_a.endswith(":301075") and key_b.endswith(":301201")

        return parse_ok and filter_ok and switch_ok and restore_ok and slot_ok
    finally:
        if saved_profiles_env is None:
            os.environ.pop("STOCK_PROFILES", None)
        else:
            os.environ["STOCK_PROFILES"] = saved_profiles_env
        config.STOCK_CODE = saved_code
        config.STOCK_NAME = saved_name
        config.STOCK_THEMES = saved_themes
        config.POSITIONS = saved_positions


def RunOilGroupChannelTest() -> bool:
    """验证石油组沪市 profile、钉钉通道切换与 group 过滤。"""
    saved_profiles_env = os.environ.get("STOCK_PROFILES")
    saved_webhook = config.DINGTALK_WEBHOOK
    saved_secret = config.DINGTALK_SECRET
    saved_oil_wh = config.DINGTALK_WEBHOOK_OIL
    saved_oil_sec = config.DINGTALK_SECRET_OIL
    saved_label = config.STOCK_GROUP_LABEL
    saved_code = config.STOCK_CODE
    try:
        config.DINGTALK_WEBHOOK = "https://hook.default/example"
        config.DINGTALK_SECRET = "SECdefault"
        config.DINGTALK_WEBHOOK_OIL = "https://hook.oil/example"
        config.DINGTALK_SECRET_OIL = "SECoil"
        os.environ["STOCK_PROFILES"] = (
            '[{"code":"301075","name":"多瑞生物","market":"sz","themes":["GLP-1"],'
            '"business":"药","positions":[],"channel":"default","group":"pharma"},'
            '{"code":"600938","name":"中国海油","market":"sh","themes":["石油","原油"],'
            '"business":"海上油气","positions":[],"channel":"oil","group":"oil_short",'
            '"group_label":"石油板块短线"}]'
        )
        profiles = config.ResolveStockProfiles()
        oil = FilterProfilesByGroup(profiles, "oil_short")
        pharma = FilterProfilesByGroup(profiles, "pharma")
        group_ok = (
            len(oil) == 1
            and oil[0].get("code") == "600938"
            and oil[0].get("market") == "sh"
            and len(pharma) == 1
        )

        webhook, secret = config.ResolveDingtalkChannel("oil")
        channel_resolve_ok = (
            webhook == "https://hook.oil/example" and secret == "SECoil"
        )

        with ApplyStockProfile(oil[0]):
            switch_ok = (
                config.STOCK_CODE == "600938"
                and config.STOCK_MARKET == "sh"
                and config.DINGTALK_WEBHOOK == "https://hook.oil/example"
                and config.DINGTALK_SECRET == "SECoil"
                and config.STOCK_GROUP_LABEL == "石油板块短线"
                and config.STOCK_GROUP == "oil_short"
                and config.STOCK_CHANNEL == "oil"
            )
        restore_ok = (
            config.DINGTALK_WEBHOOK == "https://hook.default/example"
            and config.DINGTALK_SECRET == "SECdefault"
            and config.STOCK_GROUP_LABEL == saved_label
            and config.STOCK_CODE == saved_code
        )
        return group_ok and channel_resolve_ok and switch_ok and restore_ok
    finally:
        if saved_profiles_env is None:
            os.environ.pop("STOCK_PROFILES", None)
        else:
            os.environ["STOCK_PROFILES"] = saved_profiles_env
        config.DINGTALK_WEBHOOK = saved_webhook
        config.DINGTALK_SECRET = saved_secret
        config.DINGTALK_WEBHOOK_OIL = saved_oil_wh
        config.DINGTALK_SECRET_OIL = saved_oil_sec
        config.STOCK_GROUP_LABEL = saved_label
        config.STOCK_CODE = saved_code


def RunOilShortPlayReportTest() -> bool:
    """验证石油短线实操章、强讯横幅；医药报告不含石油风险头。"""
    from dingtalk_pusher import BuildCompactDingTalkReportMarkdown
    from oil_short_playbook import OIL_SHORT_RISK_HEADER

    saved = {
        "code": config.STOCK_CODE,
        "name": config.STOCK_NAME,
        "group": config.STOCK_GROUP,
        "channel": config.STOCK_CHANNEL,
        "label": config.STOCK_GROUP_LABEL,
        "llm": config.LLM_ENABLED,
        "market": getattr(config, "STOCK_MARKET", "sz"),
    }
    try:
        config.LLM_ENABLED = False
        oil_profile = {
            "code": "600938",
            "name": "中国海油",
            "market": "sh",
            "themes": ["石油", "原油"],
            "business": "海上油气",
            "positions": [],
            "channel": "oil",
            "group": "oil_short",
            "group_label": "石油板块短线",
        }
        with ApplyStockProfile(oil_profile):
            data = {
                "realtime": {
                    **BuildMockRealtime(),
                    "price": 28.5,
                    "change_pct": 1.2,
                },
                "kline": BuildMockKline(),
                "concept": BuildMockConceptBoards(),
                "industry": BuildMockIndustryBoards(),
                "fund_flow": BuildMockFundFlow(),
            }
            analysis = AnalyzeAll(data)
            news_bundle = {
                "items": [],
                "relevant_items": [
                    {
                        "title": "布伦特原油大跌逾4%",
                        "impact": "跌",
                        "strength": 5,
                        "directness": "indirect",
                        "category": "macro",
                    },
                ],
                "news_score": -70.0,
            }
            advice = GenerateAdvice(
                analysis, data, news_items=[], news_bundle=news_bundle, session_label="盘中",
            )
            group_ok = config.STOCK_GROUP == "oil_short"
            play = advice.get("short_play") or {}
            alert = advice.get("news_alert") or {}
            play_ok = (
                play.get("entry_low")
                and play.get("stop_loss")
                and play.get("take_profit_1")
                and "position_pct" in play
            )
            # mock 强讯：覆盖为 strong_bear 验证钉钉文案
            advice["news_alert"] = {
                "level": "strong_bear",
                "score": -78,
                "headline": "布伦特大跌引发油气集体杀跌",
                "impact_on_advice": "强烈利空，优先观望止损",
            }
            advice["short_play"] = {
                **play,
                "mode": "空仓",
                "tactic": "空仓观望",
                "position_pct": 0,
                "entry_low": 27.5,
                "entry_high": 28.0,
                "stop_loss": 26.8,
                "take_profit_1": 29.0,
                "take_profit_2": 29.8,
                "hold_days": "不建议持股",
                "reasons": ["规则测试要点"],
                "discipline": ["仓位红线"],
            }
            oil_report = BuildCompactDingTalkReportMarkdown(
                data, analysis, advice, session_label="盘中",
            )
            oil_ok = (
                group_ok
                and play_ok
                and "强烈利空" in oil_report
                and "短线实操建议" in oil_report
                and OIL_SHORT_RISK_HEADER in oil_report
                and "入场区间" in oil_report
                and "止损" in oil_report
                and "止盈" in oil_report
                and "建议仓位" in oil_report
            )

        # 医药报告不含石油固定头
        pharma_profile = {
            "code": "301075",
            "name": "多瑞生物",
            "market": "sz",
            "themes": ["医药"],
            "business": "药",
            "positions": [],
            "channel": "default",
            "group": "pharma",
            "group_label": "医药双股监控",
        }
        with ApplyStockProfile(pharma_profile):
            data_p = {
                "realtime": BuildMockRealtime(),
                "kline": BuildMockKline(),
                "concept": BuildMockConceptBoards(),
                "industry": BuildMockIndustryBoards(),
                "fund_flow": BuildMockFundFlow(),
            }
            analysis_p = AnalyzeAll(data_p)
            advice_p = GenerateAdvice(
                analysis_p, data_p, news_items=[], session_label="盘中",
            )
            pharma_report = BuildCompactDingTalkReportMarkdown(
                data_p, analysis_p, advice_p, session_label="盘中",
            )
            pharma_ok = (
                OIL_SHORT_RISK_HEADER not in pharma_report
                and "短线实操建议" not in pharma_report
                and "买卖建议" in pharma_report
            )
        return oil_ok and pharma_ok
    finally:
        config.STOCK_CODE = saved["code"]
        config.STOCK_NAME = saved["name"]
        config.STOCK_GROUP = saved["group"]
        config.STOCK_CHANNEL = saved["channel"]
        config.STOCK_GROUP_LABEL = saved["label"]
        config.LLM_ENABLED = saved["llm"]
        config.STOCK_MARKET = saved["market"]


def RunCatalystNewsAndAlertTest() -> bool:
    """迪哲授权类催化应被保留，并出现在 compact「板块催化」；门控可触发。"""
    from news_alert_gate import EvaluateNewsAlert, IsInNewsWatchWindow, TryClaimNewsAlert
    from sector_catalyst_watch import (
        CollectCatalystBonus,
        IsCatalystText,
        MergeSectorWithCatalystBonus,
        TagCatalystItems,
    )
    from dingtalk_pusher import (
        BuildCompactDingTalkReportMarkdown,
        _HighlightSameDayCatalystLine,
        _IsSameDayNews,
    )
    import pandas as pd

    title = "7月14日晚间迪哲医药爆出6亿美元海外授权重磅大单"
    catalyst_ok = IsCatalystText(title, "直接引爆创新药板块早盘集体高开")

    fake_df = pd.DataFrame([
        {"标题": title, "摘要": "创新药板块集体高开，哈药股份4连板", "发布时间": "2026-07-14 20:00", "链接": ""},
        {"标题": "无关快讯足球比赛", "摘要": "NBA", "发布时间": "2026-07-14 21:00", "链接": ""},
        *[
            {"标题": f"普通医药情绪稿{i}", "摘要": "医药板块震荡", "发布时间": "2026-07-15 08:00", "链接": ""}
            for i in range(20)
        ],
    ])
    bonus = CollectCatalystBonus(fake_df, limit=6)
    bonus_ok = any("迪哲" in str(i.get("title", "")) for i in bonus)

    # 模拟普通 FilterSectorNews 只取前几条挤掉迪哲
    sector_plain = [
        {"title": f"普通医药情绪稿{i}", "content": "医药", "category": "sector"}
        for i in range(16)
    ]
    merged = MergeSectorWithCatalystBonus(sector_plain, bonus)
    merged = TagCatalystItems(merged)
    keep_ok = any("迪哲" in str(i.get("title", "")) and i.get("is_catalyst") for i in merged)

    news_bundle = {
        "items": merged,
        "relevant_items": [
            {
                "title": title,
                "content": "海外授权",
                "impact": "中性",
                "directness": "indirect",
                "category": "sector",
                "is_catalyst": True,
                "strength": 4,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            },
        ],
        "news_score": 10.0,
        "impact_stats": {"total": 1, "by_category": {"sector": 1}},
    }
    # items 列表里的催化条目强制用近时时间，避免 bonus 的「昨晚」刚好卡在 18h 边界外
    fresh_ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    for it in news_bundle["items"]:
        if it.get("is_catalyst"):
            it["time"] = fresh_ts
    news_bundle["relevant_items"][0]["time"] = fresh_ts
    verdict = EvaluateNewsAlert(news_bundle, 10.0)
    gate_ok = bool(verdict.get("worthy")) and "迪哲" in str(verdict.get("headline", ""))

    # 去重：第二次同标题应拒绝（后缀用字母，避免「数字→N」跨次碰撞）
    uniq_suffix = "".join(
        chr(97 + (int(x) % 26)) for x in str(int(datetime.now().timestamp() * 1000))[-10:]
    )
    uniq = f"testdizheuniq{uniq_suffix}"
    claim1 = TryClaimNewsAlert(uniq, group="pharma")
    claim2 = TryClaimNewsAlert(uniq, group="pharma")
    dedup_ok = claim1 and not claim2

    # 当日直接催化高亮
    today_str = date.today().strftime("%Y-%m-%d")
    same_day_item = {
        "title": "多瑞生物：获临床批件",
        "impact": "涨",
        "directness": "direct",
        "time": f"{today_str} 09:30",
        "impact_reason": "获临床批件，直接利好本公司",
        "category": "stock",
    }
    same_day_ok = _IsSameDayNews(same_day_item)
    highlight_line = _HighlightSameDayCatalystLine(
        "个股直接利好",
        "获临床批件，直接利好本公司",
        impact="涨",
        same_day=True,
    )
    style_ok = (
        "当日催化" in highlight_line
        and 'font color="#E53935"' in highlight_line
    )

    # compact 报告含板块催化
    saved_llm = config.LLM_ENABLED
    config.LLM_ENABLED = False
    try:
        data = {
            "realtime": BuildMockRealtime(),
            "kline": BuildMockKline(),
            "concept": BuildMockConceptBoards(),
            "industry": BuildMockIndustryBoards(),
            "fund_flow": BuildMockFundFlow(),
        }
        analysis = AnalyzeAll(data)
        news_bundle_today = {
            **news_bundle,
            "relevant_items": [same_day_item] + list(news_bundle.get("relevant_items") or []),
        }
        advice = GenerateAdvice(
            analysis, data, news_items=[], news_bundle=news_bundle_today, session_label="盘中",
        )
        advice["news_bundle"] = news_bundle_today
        report = BuildCompactDingTalkReportMarkdown(
            data, analysis, advice, session_label="盘中",
        )
        display_ok = (
            ("板块催化" in report or "迪哲" in report or "当日催化" in report)
            and ("当日直接催化" in report or "当日催化" in report)
            and 'font color="' in report
        )
    finally:
        config.LLM_ENABLED = saved_llm

    window_fn_ok = callable(IsInNewsWatchWindow)

    # 石油催化不与药企名单混淆
    saved_group = config.STOCK_GROUP
    try:
        config.STOCK_GROUP = "oil_short"
        oil_cat = IsCatalystText("布伦特原油大跌", "地缘缓和")
        pharma_on_oil = IsCatalystText("迪哲医药海外授权", "")
        # oil group uses oil pattern; 迪哲不一定命中油气词
        oil_branch_ok = oil_cat and not pharma_on_oil
    finally:
        config.STOCK_GROUP = saved_group

    return all([
        catalyst_ok, bonus_ok, keep_ok, gate_ok, dedup_ok,
        same_day_ok, style_ok, display_ok, window_fn_ok, oil_branch_ok,
    ])


def RunNewsWatchTitleTest() -> bool:
    """哨兵推送标题含影响预警；定时报告不含。"""
    from dingtalk_pusher import (
        BuildCompactDingTalkReportMarkdown,
        BuildPushReportTitle,
        NEWS_WATCH_ALERT_TAG,
        NEWS_WATCH_LABEL,
    )

    saved_llm = config.LLM_ENABLED
    saved_label = config.STOCK_GROUP_LABEL
    config.LLM_ENABLED = False
    try:
        data = {
            "realtime": BuildMockRealtime(),
            "kline": BuildMockKline(),
            "concept": BuildMockConceptBoards(),
            "industry": BuildMockIndustryBoards(),
            "fund_flow": BuildMockFundFlow(),
        }
        analysis = AnalyzeAll(data)
        advice = GenerateAdvice(analysis, data, news_items=[], session_label="盘中")

        watch_report = BuildCompactDingTalkReportMarkdown(
            data, analysis, advice, session_label=NEWS_WATCH_LABEL,
        )
        daily_report = BuildCompactDingTalkReportMarkdown(
            data, analysis, advice, session_label="开盘前",
        )
        watch_ok = (
            NEWS_WATCH_ALERT_TAG in watch_report
            and "可能影响个股" in watch_report
        )
        daily_ok = NEWS_WATCH_ALERT_TAG not in daily_report

        title_pharma = BuildPushReportTitle(NEWS_WATCH_LABEL)
        title_daily = BuildPushReportTitle("午盘")
        title_ok = (
            NEWS_WATCH_ALERT_TAG in title_pharma
            and "分析报告" not in title_pharma
            and NEWS_WATCH_ALERT_TAG not in title_daily
            and "午盘分析报告" in title_daily
        )

        config.STOCK_GROUP_LABEL = "石油板块短线"
        title_oil = BuildPushReportTitle(NEWS_WATCH_LABEL)
        oil_ok = (
            title_oil.startswith("石油板块短线｜")
            and NEWS_WATCH_ALERT_TAG in title_oil
        )
        config.STOCK_GROUP_LABEL = saved_label
        return watch_ok and daily_ok and title_ok and oil_ok
    finally:
        config.LLM_ENABLED = saved_llm
        config.STOCK_GROUP_LABEL = saved_label


def RunNewsFreshnessAndDedupTest() -> bool:
    """旧闻/无时间拒推；近 18h 可推；群级归一化去重与跨群隔离。"""
    from news_alert_gate import (
        EvaluateNewsAlert,
        IsNewsFreshEnough,
        NormalizeAlertTitle,
        TryClaimNewsAlert,
    )

    now = datetime.now()
    fresh_ts = (now - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M")
    stale_ts = (now - timedelta(hours=40)).strftime("%Y-%m-%d %H:%M")

    fresh_item = {
        "title": "迪哲医药爆出6亿美元海外授权重磅大单",
        "content": "创新药板块集体高开",
        "impact": "中性",
        "directness": "indirect",
        "category": "sector",
        "is_catalyst": True,
        "strength": 4,
        "time": fresh_ts,
    }
    stale_item = {
        **fresh_item,
        "time": stale_ts,
        "title": "两天前迪哲医药海外授权旧闻",
    }
    no_time_item = {k: v for k, v in fresh_item.items() if k != "time"}
    no_time_item["title"] = "无发布时间的催化稿"

    fresh_ok = IsNewsFreshEnough(fresh_item)
    stale_ok = not IsNewsFreshEnough(stale_item)
    no_time_ok = not IsNewsFreshEnough(no_time_item)

    fresh_verdict = EvaluateNewsAlert(
        {"items": [fresh_item], "relevant_items": [fresh_item]}, 10.0,
    )
    stale_verdict = EvaluateNewsAlert(
        {"items": [stale_item], "relevant_items": [stale_item]}, 10.0,
    )
    no_time_verdict = EvaluateNewsAlert(
        {"items": [no_time_item], "relevant_items": [no_time_item]}, 10.0,
    )
    eval_ok = (
        bool(fresh_verdict.get("worthy"))
        and not stale_verdict.get("worthy")
        and stale_verdict.get("reason") in ("stale_news", "no_fresh_catalyst")
        and not no_time_verdict.get("worthy")
    )

    norm_a = NormalizeAlertTitle("迪哲医药，海外授权！！")
    norm_b = NormalizeAlertTitle("迪哲医药 海外授权")
    norm_ok = norm_a == norm_b and bool(norm_a)

    # 用字母后缀避免「数字→N」后与历史测试键碰撞
    suffix = "".join(chr(97 + (int(x) % 26)) for x in str(int(now.timestamp() * 1000))[-8:])
    base = f"迪哲医药海外授权去重校验{suffix}"
    claim1 = TryClaimNewsAlert(f"{base}！！", group="pharma")
    claim2 = TryClaimNewsAlert(f"{base}  ", group="pharma")
    claim3 = TryClaimNewsAlert(f"{base}。", group="pharma")
    dedup_norm_ok = claim1 and not claim2 and not claim3

    # 同标题不同群可各推一次；同群第二次拒绝；批量认领同事件变体
    cross = f"跨群去重校验{suffix}"
    cross_pharma = TryClaimNewsAlert(cross, group="pharma")
    cross_oil = TryClaimNewsAlert(cross, group="oil_short")
    cross_pharma2 = TryClaimNewsAlert(cross, group="pharma")
    cross_group_ok = cross_pharma and cross_oil and not cross_pharma2

    from news_alert_gate import TryClaimNewsAlertBundle
    bundle_base = f"批量认领校验{suffix}"
    bundle1 = TryClaimNewsAlertBundle(
        [f"{bundle_base}甲", f"{bundle_base}乙"], group="pharma",
    )
    bundle2 = TryClaimNewsAlertBundle(
        [f"{bundle_base}乙变体", f"{bundle_base}乙"], group="pharma",
    )
    bundle_ok = bundle1 and not bundle2

    stale_direct = {
        "title": "公司获得药品注册证书",
        "content": "利好",
        "impact": "涨",
        "directness": "direct",
        "relevant": True,
        "category": "stock",
        "strength": 5,
        "time": stale_ts,
    }
    direct_verdict = EvaluateNewsAlert(
        {"relevant_items": [stale_direct], "items": [stale_direct]}, 20.0,
    )
    stale_direct_ok = (
        not direct_verdict.get("worthy")
        and direct_verdict.get("reason") == "stale_news"
    )

    return all([
        fresh_ok, stale_ok, no_time_ok, eval_ok,
        norm_ok, dedup_norm_ok, cross_group_ok, bundle_ok, stale_direct_ok,
    ])


def RunNewsWatchWindowTest() -> bool:
    """资讯哨兵四定点窗口：每天 09:00/12:30/18:00/22:00 ±10 分钟，含周末。"""
    from news_alert_gate import IsInNewsWatchWindow

    sat_morning = datetime(2026, 7, 18, 9, 0)   # 周六
    weekday_night = datetime(2026, 7, 15, 22, 0)  # 周三
    weekday_noon = datetime(2026, 7, 15, 12, 30)
    weekday_evening = datetime(2026, 7, 15, 18, 0)
    weekday_outside = datetime(2026, 7, 15, 15, 0)
    weekday_before = datetime(2026, 7, 15, 8, 40)  # 距 09:00 槽 >10 分钟
    slot_edge = datetime(2026, 7, 15, 9, 10)   # 09:00 +10 分钟边界
    slot_outside = datetime(2026, 7, 15, 9, 11)  # 超出 ±10

    return all([
        IsInNewsWatchWindow(sat_morning),
        IsInNewsWatchWindow(weekday_night),
        IsInNewsWatchWindow(weekday_noon),
        IsInNewsWatchWindow(weekday_evening),
        not IsInNewsWatchWindow(weekday_outside),
        not IsInNewsWatchWindow(weekday_before),
        IsInNewsWatchWindow(slot_edge),
        not IsInNewsWatchWindow(slot_outside),
    ])


def RunCatalystLlmDisplayTest() -> bool:
    """哨兵报告展示 AI 催化解读、发布时间；规则兜底不含占位语。"""
    from dingtalk_pusher import (
        BuildCompactDingTalkReportMarkdown,
        NEWS_WATCH_LABEL,
        _FormatNewsTime,
        _ItemDisplayReason,
    )
    from news_analyzer import _ApplyRuleFallback, _RuleFallbackImpactReason

    saved_llm = config.LLM_ENABLED
    config.LLM_ENABLED = False
    try:
        data = {
            "realtime": BuildMockRealtime(),
            "kline": BuildMockKline(),
            "concept": BuildMockConceptBoards(),
            "industry": BuildMockIndustryBoards(),
            "fund_flow": BuildMockFundFlow(),
        }
        analysis = AnalyzeAll(data)

        publish_raw = "2026-07-14 20:00"
        publish_short = _FormatNewsTime(publish_raw)
        catalyst_item = {
            "title": "迪哲医药6亿美元海外授权大单引爆创新药板块",
            "content": "创新药板块集体高开",
            "impact": "中性",
            "directness": "indirect",
            "category": "sector",
            "is_catalyst": True,
            "strength": 4,
            "impact_reason": "板块情绪传导，对本公司间接影响",
            "time": publish_raw,
        }
        news_bundle = {
            "relevant_items": [catalyst_item],
            "scored_items": [catalyst_item],
            "impact_stats": {"total": 1, "by_category": {"sector": 1}},
            "analysis_source": "ai",
            "catalyst_llm": {
                "catalyst_summary": (
                    "迪哲海外授权大单提振创新药板块情绪，"
                    "多瑞生物主业为补液注射液，传导以板块跟风为主，勿当个股直接订单。"
                ),
                "impact_on_stock": "涨",
                "action_hint": "板块情绪利好，勿当个股直接订单",
                "trigger_title": catalyst_item["title"][:40],
                "trigger_time": publish_raw,
            },
        }
        advice = GenerateAdvice(
            analysis, data, news_items=[], news_bundle=news_bundle,
            session_label=NEWS_WATCH_LABEL,
        )
        advice["news_bundle"] = news_bundle
        report = BuildCompactDingTalkReportMarkdown(
            data, analysis, advice, session_label=NEWS_WATCH_LABEL,
        )
        llm_display_ok = (
            "AI催化解读" in report
            and "对本股影响" in report
            and "操作建议" in report
            and "勿当个股直接订单" in report
            and "资讯发布" in report
            and "触发资讯" in report
            and publish_short in report
            and f"`{publish_short}`" in report
            and ("板块催化" in report or "迪哲" in report)
        )

        rule_items = [
            {
                "title": "医药板块震荡走强，资金关注创新药",
                "content": "板块普涨",
                "category": "sector",
                "is_catalyst": True,
            },
        ]
        rule_reason = _RuleFallbackImpactReason(rule_items[0])
        rule_scored = _ApplyRuleFallback(rule_items, data)
        rule_ok = (
            "建议启用 AI" not in rule_reason
            and "规则兜底" not in rule_reason
            and all(
                "建议启用 AI" not in str(i.get("impact_reason", ""))
                for i in rule_scored
            )
        )
        display_reason_ok = (
            _ItemDisplayReason({
                "impact_reason": "规则兜底，建议启用 AI 获取精准影响",
                "title": "测试标题摘要",
            }) == "测试标题摘要"
        )
        return llm_display_ok and rule_ok and display_reason_ok
    finally:
        config.LLM_ENABLED = saved_llm


def RunNewsPrefetchTest() -> bool:
    """哨兵 prefetch 复用 bundle 时不重复 FetchNews。"""
    from unittest.mock import patch

    from main import _JobForCurrentProfile

    fetch_count = {"n": 0}

    def _CountFetch(*_args, **_kwargs):
        fetch_count["n"] += 1
        return BuildMockNews()

    def _FakeFetchAll():
        return {
            "realtime": BuildMockRealtime(),
            "kline": BuildMockKline(),
            "concept": BuildMockConceptBoards(),
            "industry": BuildMockIndustryBoards(),
            "fund_flow": BuildMockFundFlow(),
        }

    def _FakeAnalyze(bundle, data):
        bundle = dict(bundle)
        bundle["analysis_source"] = "rule"
        bundle["relevant_items"] = list(bundle.get("items") or [])[:2]
        return bundle

    def _FakeEnrich(bundle, data, trigger_headline=""):
        return bundle

    def _FakeAnalyzeAll(_data):
        return {"direction": "震荡", "tech_score": 0, "fund_score": 0,
                "sector_score": 0, "news_score": 0, "weighted_score": 0,
                "indicators": {"price": 61.67}}

    def _FakeMerge(analysis, _bundle):
        return analysis

    def _FakeAdvice(*_args, **_kwargs):
        return {"buy_ok": False, "sell_ok": False, "llm_used": False}

    def _FakePush(*_args, **_kwargs):
        return None

    saved_llm = config.LLM_ENABLED
    config.LLM_ENABLED = False
    try:
        data = _FakeFetchAll()
        bundle = _FakeAnalyze(BuildMockNews(), data)
        prefetch = {"data": data, "news_bundle": bundle}

        with patch("main.FetchNews", side_effect=_CountFetch), \
             patch("main.FetchAllData", side_effect=_FakeFetchAll), \
             patch("main.AnalyzeNewsImpact", side_effect=_FakeAnalyze), \
             patch("main.EnrichCatalystWithLlm", side_effect=_FakeEnrich), \
             patch("main.AnalyzeAll", side_effect=_FakeAnalyzeAll), \
             patch("main.MergeNewsIntoAnalysis", side_effect=_FakeMerge), \
             patch("main.GenerateAdvice", side_effect=_FakeAdvice), \
             patch("main.PushReport", side_effect=_FakePush), \
             patch("main.ResolveSessionAndHorizon", return_value=("盘中", {})):
            _JobForCurrentProfile(
                session_label="资讯速报",
                force_run=True,
                prefetch=prefetch,
                trigger_headline="测试触发",
            )
        return fetch_count["n"] == 0
    finally:
        config.LLM_ENABLED = saved_llm


def RunThemeMismatchTest() -> bool:
    """验证板块/政策 GLP-1 资讯不会被误标为个股利好。"""
    saved_themes = list(config.STOCK_THEMES)
    saved_business = config.STOCK_BUSINESS
    saved_llm = config.LLM_ENABLED
    try:
        config.STOCK_THEMES = ["GLP-1", "司美格鲁肽", "多肽原料药"]
        config.STOCK_BUSINESS = (
            "主营补液注射液；多肽/GLP-1业务刚起步、收入规模小、无海外长协大单"
        )
        config.LLM_ENABLED = False

        data = {
            "realtime": BuildMockRealtime(),
            "kline": BuildMockKline(),
            "concept": BuildMockConceptBoards(),
            "industry": BuildMockIndustryBoards(),
            "fund_flow": BuildMockFundFlow(),
        }
        news_bundle = AnalyzeNewsImpact(BuildMockNews(), data)
        relevant = list(news_bundle.get("relevant_items") or [])

        policy_glp1 = [
            i for i in relevant
            if i.get("category") == "policy" and "基药目录" in str(i.get("title", ""))
        ]
        web_glp1 = [
            i for i in relevant
            if i.get("category") == "web_search" and "GLP-1" in str(i.get("title", ""))
        ]
        sector_items = [i for i in relevant if i.get("category") == "sector"]

        mismatch_ok = all(
            str(i.get("directness", "")) in ("indirect", "mismatch")
            and i.get("impact") != "涨"
            for i in policy_glp1 + web_glp1 + sector_items
        )

        news_score, _ = CalcNewsScore(relevant)
        score_ok = news_score <= 5

        advice = GenerateAdvice(
            AnalyzeAll(data),
            data,
            news_items=news_bundle["items"],
            news_bundle=news_bundle,
            session_label="收盘后",
            mode="daily",
            prediction_horizon=ResolvePredictionHorizon("收盘后"),
        )
        advice["news_summary"] = "医药板块活跃，但个股主业与领涨方向存在分化"
        report = BuildDingTalkReportMarkdown(
            data, AnalyzeAll(data), advice,
            session_label="收盘后",
            report_mode="daily",
        )
        no_false_bullish = (
            "利好** GLP-1" not in report
            and "利好** 产业链受益" not in report
            and (
                "产业链受益" not in report.split("资讯研判")[1].split("买卖建议")[0]
                if "资讯研判" in report and "买卖建议" in report
                else "产业链受益" not in report
            )
        )
        background_ok = bool(GetBackgroundNewsItems(relevant)) or bool(
            i for i in relevant if str(i.get("directness", "")) == "mismatch"
        )

        return mismatch_ok and score_ok and no_false_bullish and background_ok
    finally:
        config.STOCK_THEMES = saved_themes
        config.STOCK_BUSINESS = saved_business
        config.LLM_ENABLED = saved_llm


def RunCompactModeTest(report: str) -> bool:
    """验证精简版钉钉报告结构。"""
    news_idx = report.find("、资讯研判")
    trade_idx = report.find("、买卖建议")
    move_idx = report.find("、涨跌归因")
    return (
        "分层预测" in report
        and "、资讯研判" in report
        and "盘面信号" in report
        and "、涨跌归因" in report
        and "综合结论" in report
        and "四维评分" in report
        and "个股直接利好" in report
        and "详细依据" not in report
        and "资讯解读" not in report
        and "【个股资讯】" not in report
        and news_idx >= 0
        and trade_idx > news_idx
        and move_idx > trade_idx
    )


def RunFullModeTest(
    data: dict[str, Any],
    analysis: dict[str, Any],
    advice: dict[str, Any],
) -> bool:
    """验证 full 模式仍输出完整长报告。"""
    saved_mode = config.DINGTALK_REPORT_MODE
    try:
        config.DINGTALK_REPORT_MODE = "full"
        full_report = BuildDingTalkReportMarkdown(
            data, analysis, advice,
            session_label="收盘后",
            report_mode="daily",
        )
        price_move = advice.get("price_move") or {}
        return (
            "综合预测" in full_report
            and "详细依据" in full_report
            and "资讯解读" in full_report
            and "涨跌归因拆解" in full_report
            and "【个股资讯】" in full_report
            and bool(price_move.get("dimensions"))
        )
    finally:
        config.DINGTALK_REPORT_MODE = saved_mode


def RunMockTest() -> None:
    """运行模拟测试，打印完整 Markdown 报告。"""
    print("=" * 60)
    print("模拟数据测试 — 无需网络")
    print("=" * 60)

    config.POSITIONS = [
        {"account": "账户A", "cost": 167.955, "shares": 100},
        {"account": "账户B", "cost": 63.38, "shares": 800},
    ]
    config.LLM_ENABLED = False
    config.STOCK_THEMES = ["GLP-1", "司美格鲁肽", "多肽原料药"]
    config.STOCK_BUSINESS = (
        "主营补液注射液；多肽/GLP-1业务刚起步、收入规模小、无海外长协大单"
    )
    config.PRICE_MOVE_MIN_PCT = 2.0
    config.DINGTALK_REPORT_MODE = "compact"

    data = {
        "realtime": BuildMockRealtime(),
        "kline": BuildMockKline(),
        "concept": BuildMockConceptBoards(),
        "industry": BuildMockIndustryBoards(),
        "fund_flow": BuildMockFundFlow(),
        "lhb": None,
        "margin": {
            "margin_balance": 12500000,
            "margin_balance_prev": 12100000,
            "margin_balance_change": 400000,
            "margin_buy": 850000,
            "trade_date": "2026-07-09",
        },
    }

    news_bundle = AnalyzeNewsImpact(BuildMockNews(), data)
    for item in news_bundle.get("relevant_items", []):
        title = str(item.get("title", ""))
        if item.get("category") == "stock" and "注册证书" in title:
            item["impact"] = "涨"
            item["directness"] = "direct"
            item["strength"] = 4
            item["impact_reason"] = "子公司获药品注册证书，直接利好本公司"
    news_score, news_signals = CalcNewsScore(news_bundle.get("relevant_items", []))
    news_bundle["news_score"] = news_score
    news_bundle["news_signals"] = news_signals

    analysis = AnalyzeAll(data)
    base_weighted = analysis["weighted_score"]
    analysis = MergeNewsIntoAnalysis(analysis, news_bundle)
    advice = GenerateAdvice(
        analysis,
        data,
        news_items=news_bundle["items"],
        news_bundle=news_bundle,
        session_label="收盘后",
        mode="daily",
        prediction_horizon=ResolvePredictionHorizon("收盘后"),
    )

    advice["news_summary"] = "子公司获注册证书为个股直接利好；医药板块活跃但主线与本公司主业存在分化"
    advice["policy_impact"] = "基药目录调整利好行业龙头，本公司补液主业传导有限"
    advice["theme_mismatch_note"] = "今日领涨中药/家用医疗，与补液注射液主业赛道割裂"

    report = BuildDingTalkReportMarkdown(
        data, analysis, advice,
        session_label="收盘后",
        report_mode="daily",
    )

    print(report)
    print("=" * 60)

    price = data["realtime"]["price"]
    buy_ok = advice["buy_high"] < price < advice["sell_low"]
    has_portfolio_section = "持仓与回本" in report
    stats = news_bundle.get("impact_stats", {})
    by_cat = stats.get("by_category", {})
    categories_ok = all(by_cat.get(k, 0) > 0 for k in ("stock", "sector", "policy", "macro"))
    compact_ok = RunCompactModeTest(report)
    full_mode_ok = RunFullModeTest(data, analysis, advice)
    print(f"定价校验: 买入<{price}<卖出 → {'通过' if buy_ok else '失败'}")
    print(f"日常报告含持仓章节: {'否（正确）' if not has_portfolio_section else '是（错误）'}")
    print(f"相关资讯: {stats}")
    print(f"精简版章节结构: {'通过' if compact_ok else '失败'}")
    print(f"完整版模式(full): {'通过' if full_mode_ok else '失败'}")
    print(f"四维分类计数: {'通过' if categories_ok else '失败'} — {by_cat}")
    news_merge_ok = (
        analysis.get("news_score", 0) != 0
        and analysis.get("weighted_score") != base_weighted
        and "资讯研判" in report
        and "四维评分" in report
    )
    print(f"资讯面融合: {'通过' if news_merge_ok else '失败'} — news_score={analysis.get('news_score')}, weighted={analysis.get('weighted_score'):+.1f}")
    trade_format_ok = (
        "风险提示" in report
        and "第一加仓（优先等待）" in report
        and "第二加仓（次选）" in report
        and "第一卖出（先减）" in report
        and "第二卖出（再减）" in report
        and "不宜加仓" in report
        and "加仓纪律" in report
        and "推荐价位" in report
        and "定价说明" in report
        and "AI定价无效" not in report
    )
    zones = advice.get("trade_zones") or {}
    plan = advice.get("trade_plan") or {}
    at1 = zones.get("add_tier1") or plan.get("add_tier1", {})
    at2 = zones.get("add_tier2") or plan.get("add_tier2", {})
    st1 = zones.get("sell_tier1") or plan.get("sell_tier1", {})
    tier_order_ok = (
        at1.get("low", 0) < at2.get("low", 0)
        and at2.get("high", 0) < price < st1.get("low", 999)
    )
    levels = ResolveTradeDisplayLevels(advice)
    levels_ok = (
        abs(levels["s1"] - levels["buy_high"]) < 0.01
        and abs(levels["s2"] - levels["buy_low"]) < 0.01
        and abs(levels["r1"] - levels["sell_low"]) < 0.01
        and abs(levels["r2"] - levels["sell_high"]) < 0.01
        and levels["stop_loss"] < levels["s2"]
        and tier_order_ok
    )
    print(f"买卖建议排版: {'通过' if trade_format_ok else '失败'}")
    print(f"分档顺序: {'通过' if tier_order_ok else '失败'} — "
          f"加一={at1.get('low')}~{at1.get('high')} 加二={at2.get('low')}~{at2.get('high')}")
    print(f"支撑压力一致性: {'通过' if levels_ok else '失败'} — S1={levels['s1']} S2={levels['s2']} stop={levels['stop_loss']}")
    layered_ok = (
        "分层预测" in report
        and VerifyLayeredPredictions(report, "明日预测")
        and advice.get("predictions", {}).get("near_term")
        and advice.get("predictions", {}).get("short_term")
        and advice.get("predictions", {}).get("long_term")
    )
    print(f"分层综合预测: {'通过' if layered_ok else '失败'}")

    report_pre, _ = RunPredictionHorizonTest(
        data, analysis, news_bundle, "开盘前",
    )
    pre_ok = VerifyLayeredPredictions(report_pre, "明日预测")
    report_intraday, _ = RunPredictionHorizonTest(
        data, analysis, news_bundle, "下午盘中",
    )
    intraday_ok = VerifyLayeredPredictions(report_intraday, "今日预测")
    print(f"开盘前Horizon: {'通过' if pre_ok else '失败'}")
    print(f"盘中Horizon: {'通过' if intraday_ok else '失败'}")

    sat_ok, fri_ok, sat_dingtalk_ok = RunHorizonResolverTests(data, analysis, news_bundle)
    print(f"周末Horizon解析: {'通过' if sat_ok else '失败'}")
    print(f"周五Horizon解析: {'通过' if fri_ok else '失败'}")
    print(f"周末钉钉资讯融合: {'通过' if sat_dingtalk_ok else '失败'}")

    kline_merge_ok = RunKlineMergeTest()
    tail_horizon_ok = RunTailHorizonTest()
    print(f"盘中K线融合: {'通过' if kline_merge_ok else '失败'}")
    print(f"尾盘Horizon: {'通过' if tail_horizon_ok else '失败'}")
    theme_mismatch_ok = RunThemeMismatchTest()
    print(f"题材错配分层: {'通过' if theme_mismatch_ok else '失败'}")

    trading_day_date_ok = RunTradingDayDateLabelTest()
    print(f"涨跌归因日期标注: {'通过' if trading_day_date_ok else '失败'}")

    multi_stock_ok = RunMultiStockProfileTest()
    print(f"多股监控 Profile: {'通过' if multi_stock_ok else '失败'}")

    oil_group_ok = RunOilGroupChannelTest()
    print(f"石油双群通道: {'通过' if oil_group_ok else '失败'}")

    oil_short_play_ok = RunOilShortPlayReportTest()
    print(f"石油短线实操报告: {'通过' if oil_short_play_ok else '失败'}")

    catalyst_alert_ok = RunCatalystNewsAndAlertTest()
    print(f"板块催化与资讯门控: {'通过' if catalyst_alert_ok else '失败'}")

    news_watch_title_ok = RunNewsWatchTitleTest()
    print(f"资讯哨兵标题区分: {'通过' if news_watch_title_ok else '失败'}")

    fresh_dedup_ok = RunNewsFreshnessAndDedupTest()
    print(f"哨兵新鲜度与去重: {'通过' if fresh_dedup_ok else '失败'}")

    news_watch_window_ok = RunNewsWatchWindowTest()
    print(f"资讯哨兵四槽窗口: {'通过' if news_watch_window_ok else '失败'}")

    catalyst_llm_ok = RunCatalystLlmDisplayTest()
    print(f"AI催化解读展示: {'通过' if catalyst_llm_ok else '失败'}")

    news_prefetch_ok = RunNewsPrefetchTest()
    print(f"哨兵 prefetch 复用: {'通过' if news_prefetch_ok else '失败'}")

    news_judge_ok = (
        "资讯研判" in report
        and "个股直接利好" in report
        and (
            "产业链受益" not in report.split("资讯研判")[1].split("买卖建议")[0]
            if "资讯研判" in report and "买卖建议" in report
            else "产业链受益" not in report
        )
    )
    print(f"钉钉资讯研判块: {'通过' if news_judge_ok else '失败'}")

    if (
        not categories_ok or not news_merge_ok
        or not trade_format_ok or not levels_ok or not layered_ok
        or not pre_ok or not intraday_ok
        or not sat_ok or not fri_ok or not sat_dingtalk_ok
        or not compact_ok or not full_mode_ok
        or not kline_merge_ok or not tail_horizon_ok
        or not theme_mismatch_ok or not trading_day_date_ok
        or not multi_stock_ok or not oil_group_ok
        or not oil_short_play_ok or not catalyst_alert_ok
        or not news_watch_title_ok or not fresh_dedup_ok
        or not news_watch_window_ok
        or not catalyst_llm_ok
        or not news_prefetch_ok or not news_judge_ok
    ):
        raise SystemExit(1)

    # portfolio_advice = GeneratePortfolioAdvice(data, analysis)
    # print("\n--- 持仓专报预览 ---")
    # print(BuildPortfolioReportMarkdown(portfolio_advice))
    print("=" * 60)
    print("模拟测试完成")


if __name__ == "__main__":
    RunMockTest()
