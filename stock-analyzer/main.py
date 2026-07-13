"""主入口 — 定时调度 + 手动实时分析 + 持仓建议。"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime

import schedule

import config
from advisor import GenerateAdvice
from analyzer import AnalyzeAll, MergeNewsIntoAnalysis
from data_fetcher import FetchAllData
from dingtalk_pusher import PushErrorNotice, PushPortfolioReport, PushReport
from market_context import ResolveSessionAndHorizon
from news_analyzer import AnalyzeNewsImpact
from news_fetcher import FetchNews, FlattenNewsItems
from portfolio_advisor import GeneratePortfolioAdvice
from utils import NowStr, SetupLogger

logger = SetupLogger(config.LOG_LEVEL)

_SESSION_LABELS: dict[str, str] = {
    "09:00": "开盘前",
    "11:30": "午盘",
    "14:30": "尾盘",
}


def ParseArgs() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="股票分析钉钉推送 — 支持定时调度与手动实时分析",
    )
    parser.add_argument(
        "--now", "-n",
        action="store_true",
        help="立即执行一次日常分析（无持仓内容）",
    )
    parser.add_argument(
        "--portfolio", "-p",
        action="store_true",
        help="立即生成分账户持仓建议并推送",
    )
    parser.add_argument(
        "--scheduled",
        action="store_true",
        help="执行一次日常推送（供 GitHub Actions 等外部调度，含非交易日）",
    )
    parser.add_argument(
        "--push-time",
        type=str,
        default=None,
        help="指定推送时刻 HH:MM（如 09:00），用于映射开盘前/午盘/尾盘标签",
    )
    return parser.parse_args()


def GetSessionLabel(push_time: str | None = None) -> str:
    """根据推送时间或当前时刻返回时段标签。"""
    if push_time and push_time in _SESSION_LABELS:
        return _SESSION_LABELS[push_time]

    now = datetime.now()
    minutes = now.hour * 60 + now.minute
    if minutes < 9 * 60 + 30:
        return "开盘前"
    if minutes < 11 * 60 + 30:
        return "上午盘中"
    if minutes < 13 * 60:
        return "午盘"
    if minutes < 14 * 60:
        return "下午盘中"
    if minutes < 15 * 60:
        return "尾盘"
    return "收盘后"


def Job(
    session_label: str | None = None,
    push_time: str | None = None,
    force_run: bool = False,
) -> None:
    """执行一次日常实时分析推送（不含持仓）。"""
    label = session_label or GetSessionLabel(push_time)
    run_mode = "手动实时" if force_run else "定时"
    logger.info("=" * 50)
    logger.info(
        "开始执行 %s(%s) %s日常分析 — 时段:%s ｜ %s",
        config.STOCK_NAME,
        config.STOCK_CODE,
        run_mode,
        label,
        NowStr(),
    )

    try:
        logger.info("正在拉取实时行情与 K 线...")
        data = FetchAllData()

        logger.info("正在并行拉取多源资讯（个股/板块/政策/宏观/联网）...")
        news_bundle = FetchNews(data)
        news_bundle = AnalyzeNewsImpact(news_bundle, data)
        news_items = FlattenNewsItems(news_bundle)
        stats = news_bundle.get("impact_stats", {})
        by_category = stats.get("by_category", {})
        logger.info(
            "资讯就绪 — 个股:%s 板块:%s 政策:%s 宏观:%s 联网:%s ｜ 相关:%s 涨/跌:%s",
            by_category.get("stock", 0),
            by_category.get("sector", 0),
            by_category.get("policy", 0),
            by_category.get("macro", 0),
            by_category.get("web_search", 0),
            stats.get("relevant", 0),
            stats.get("impactful", 0),
        )

        if data.get("realtime") is None and data.get("kline") is None:
            logger.error("核心数据（实时行情+K线）均获取失败")
            PushErrorNotice()
            return

        analysis = AnalyzeAll(data)
        analysis = MergeNewsIntoAnalysis(analysis, news_bundle)
        display_label, horizon = ResolveSessionAndHorizon(label)
        advice = GenerateAdvice(
            analysis,
            data,
            news_items=news_items,
            news_bundle=news_bundle,
            session_label=display_label,
            mode="daily",
            prediction_horizon=horizon,
        )
        success = PushReport(
            data, analysis, advice,
            session_label=display_label,
            report_mode="daily",
            push_time=push_time,
        )

        if success:
            logger.info("日常分析完成，报告已推送")
        else:
            logger.warning("日常分析完成，但钉钉推送未成功")

    except Exception as exc:
        logger.exception("任务执行异常: %s", exc)
        PushErrorNotice(f"任务执行异常: {exc}")

    logger.info("=" * 50)


def JobPortfolio(force_run: bool = True) -> None:
    """执行分账户持仓建议推送。"""
    logger.info("=" * 50)
    logger.info(
        "开始执行 %s(%s) 分账户持仓建议 — %s",
        config.STOCK_NAME,
        config.STOCK_CODE,
        NowStr(),
    )

    try:
        if not config.POSITIONS:
            logger.error("未配置 POSITIONS，无法生成持仓建议")
            PushErrorNotice("未配置持仓信息，请在 .env 中设置 POSITIONS")
            return

        data = FetchAllData()
        if data.get("realtime") is None and data.get("kline") is None:
            logger.error("核心数据获取失败")
            PushErrorNotice()
            return

        analysis = AnalyzeAll(data)
        portfolio_advice = GeneratePortfolioAdvice(data, analysis)
        success = PushPortfolioReport(portfolio_advice)

        if success:
            logger.info("持仓建议已推送 — %d 个账户", len(portfolio_advice.get("accounts", [])))
        else:
            logger.warning("持仓建议生成完成，但钉钉推送未成功")

    except Exception as exc:
        logger.exception("持仓建议异常: %s", exc)
        PushErrorNotice(f"持仓建议异常: {exc}")

    logger.info("=" * 50)


def RunScheduler() -> None:
    """启动定时调度，支持多次推送。"""
    push_times = config.PUSH_TIMES
    logger.info("定时模式启动，每日推送时间点: %s", ", ".join(push_times))
    logger.info("手动: python main.py --now ｜ 持仓: python main.py --portfolio")

    for push_time in push_times:
        label = GetSessionLabel(push_time)
        schedule.every().day.at(push_time).do(Job, session_label=label, push_time=push_time)
        logger.info("已注册调度 — %s (%s)", push_time, label)

    while True:
        try:
            schedule.run_pending()
        except Exception as exc:
            logger.exception("调度循环异常: %s", exc)
        time.sleep(30)


def Main() -> None:
    """程序入口。"""
    args = ParseArgs()

    if args.portfolio:
        logger.warning("持仓专报推送已暂停，请使用日常报告")
        return
    # if args.portfolio:
    #     logger.info("股票分析工具 — %s(%s) 模式:持仓建议", config.STOCK_NAME, config.STOCK_CODE)
    #     JobPortfolio(force_run=True)
    #     return

    if args.scheduled:
        label = GetSessionLabel(args.push_time)
        logger.info(
            "股票分析工具 — %s(%s) 模式:外部调度 push_time=%s 时段=%s",
            config.STOCK_NAME,
            config.STOCK_CODE,
            args.push_time or "auto",
            label,
        )
        Job(session_label=label, push_time=args.push_time, force_run=False)
        return

    is_manual = args.now or config.MANUAL_RUN
    logger.info(
        "股票分析钉钉推送工具启动 — %s(%s) 模式:%s",
        config.STOCK_NAME,
        config.STOCK_CODE,
        "手动日常" if is_manual else "定时",
    )

    if is_manual:
        Job(force_run=True)
    else:
        RunScheduler()


if __name__ == "__main__":
    try:
        Main()
    except KeyboardInterrupt:
        logger.info("程序已停止")
        sys.exit(0)
