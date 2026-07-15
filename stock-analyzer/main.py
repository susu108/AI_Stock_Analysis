"""主入口 — 定时调度 + 手动实时分析 + 持仓建议。"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from typing import Any

import schedule

import config
from advisor import GenerateAdvice
from analyzer import AnalyzeAll, MergeNewsIntoAnalysis
from data_fetcher import FetchAllData
from dingtalk_pusher import NEWS_WATCH_LABEL, PushErrorNotice, PushPortfolioReport, PushReport
from llm_advisor import IsLlmEnabled
from market_context import ResolveSessionAndHorizon
from news_analyzer import AnalyzeNewsImpact
from news_catalyst_llm import EnrichCatalystWithLlm
from news_fetcher import FetchNews, FlattenNewsItems
from portfolio_advisor import GeneratePortfolioAdvice
from scheduler_guard import ReleasePushSlot, TryAcquireSchedulerLock, TryClaimPushSlot
from stock_profile import ApplyStockProfile, FilterProfilesByCode, FilterProfilesByGroup
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
        "--local-scheduler",
        action="store_true",
        help="显式启用本地常驻定时调度（默认关闭，请用 GitHub Actions）",
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
    parser.add_argument(
        "--stock-code",
        type=str,
        default=None,
        help="仅分析并推送指定股票代码（如 301201），默认推送 STOCK_PROFILES 中全部",
    )
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        help="仅推送指定监控组（如 pharma / oil_short），默认全部",
    )
    parser.add_argument(
        "--news-watch",
        action="store_true",
        help="资讯哨兵：轻量拉资讯，命中强催化后再推送完整报告",
    )
    parser.add_argument(
        "--ignore-watch-window",
        action="store_true",
        help="news-watch 时忽略交易时段窗口限制（测试用）",
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
    stock_code: str | None = None,
    group: str | None = None,
) -> None:
    """执行一次日常实时分析推送（不含持仓），支持多股 profile 循环。"""
    profiles = FilterProfilesByGroup(config.ResolveStockProfiles(), group)
    profiles = FilterProfilesByCode(profiles, stock_code)
    if group and not profiles:
        logger.error("未找到监控组 %s 的 profile 配置", group)
        return
    if stock_code and not profiles:
        logger.error("未找到股票代码 %s 的 profile 配置", stock_code)
        PushErrorNotice(f"未找到股票 {stock_code} 的监控配置")
        return

    for profile in profiles:
        with ApplyStockProfile(profile):
            _JobForCurrentProfile(
                session_label=session_label,
                push_time=push_time,
                force_run=force_run,
            )


def _JobForCurrentProfile(
    session_label: str | None = None,
    push_time: str | None = None,
    force_run: bool = False,
    prefetch: dict[str, Any] | None = None,
    trigger_headline: str = "",
) -> None:
    """对当前 config 中的股票执行一次日常分析推送。"""
    label = session_label or GetSessionLabel(push_time)
    run_mode = "手动实时" if force_run else "定时"
    group_tag = config.STOCK_GROUP_LABEL or ""
    prefix = f"[{group_tag}/{config.STOCK_CODE}] " if group_tag else f"[{config.STOCK_CODE}] "
    logger.info("=" * 50)
    logger.info(
        "%s开始执行 %s(%s) %s日常分析 — 时段:%s ｜ %s",
        prefix,
        config.STOCK_NAME,
        config.STOCK_CODE,
        run_mode,
        label,
        NowStr(),
    )

    try:
        pref = prefetch or {}
        pref_data = pref.get("data")
        pref_bundle = pref.get("news_bundle")

        if pref_data is not None:
            logger.info("复用哨兵预拉行情数据")
            data = pref_data
        else:
            logger.info("正在拉取实时行情与 K 线...")
            data = FetchAllData()

        if pref_bundle is not None:
            logger.info("复用哨兵已分析资讯包（跳过 FetchNews）")
            news_bundle = dict(pref_bundle)
        else:
            logger.info("正在并行拉取多源资讯（个股/板块/政策/宏观/联网）...")
            news_bundle = FetchNews(data)
            news_bundle = AnalyzeNewsImpact(news_bundle, data)

        news_bundle = EnrichCatalystWithLlm(
            news_bundle, data, trigger_headline=trigger_headline,
        )
        news_items = FlattenNewsItems(news_bundle)
        stats = news_bundle.get("impact_stats", {})
        by_category = stats.get("by_category", {})
        logger.info(
            "资讯就绪 — 个股:%s 板块:%s 政策:%s 宏观:%s 联网:%s ｜ 相关:%s 涨/跌:%s ｜ 来源:%s",
            by_category.get("stock", 0),
            by_category.get("sector", 0),
            by_category.get("policy", 0),
            by_category.get("macro", 0),
            by_category.get("web_search", 0),
            stats.get("relevant", 0),
            stats.get("impactful", 0),
            news_bundle.get("analysis_source", "?"),
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

        if not force_run and not TryClaimPushSlot(
            push_time, display_label, stock_code=config.STOCK_CODE,
        ):
            logger.info("本时段推送已由其他实例完成，跳过")
            return

        success = PushReport(
            data, analysis, advice,
            session_label=display_label,
            report_mode="daily",
            push_time=push_time,
        )

        if not success and not force_run:
            ReleasePushSlot(push_time, display_label, stock_code=config.STOCK_CODE)

        if success:
            logger.info(
                "%s(%s) 日常分析完成，报告已推送",
                config.STOCK_NAME,
                config.STOCK_CODE,
            )
        else:
            logger.warning(
                "%s(%s) 日常分析完成，但钉钉推送未成功",
                config.STOCK_NAME,
                config.STOCK_CODE,
            )

    except Exception as exc:
        logger.exception("%s(%s) 任务执行异常: %s", config.STOCK_NAME, config.STOCK_CODE, exc)
        PushErrorNotice(f"{config.STOCK_NAME}({config.STOCK_CODE}) 任务执行异常: {exc}")

    logger.info("=" * 50)


def JobNewsWatch(
    group: str | None = None,
    stock_code: str | None = None,
    *,
    ignore_window: bool = False,
) -> int:
    """资讯哨兵：组内逐股轻量拉资讯，过门控且未去重则 force 推送。"""
    from news_alert_gate import (
        EvaluateNewsAlert,
        IsInNewsWatchWindow,
        TryClaimNewsAlert,
    )

    llm_status = "enabled" if IsLlmEnabled() else "disabled"
    logger.info("news-watch: LLM 资讯分析 %s", llm_status)

    if not ignore_window and not IsInNewsWatchWindow():
        logger.info("news-watch: 当前不在哨兵窗口，skip")
        return 0

    profiles = FilterProfilesByGroup(config.ResolveStockProfiles(), group)
    profiles = FilterProfilesByCode(profiles, stock_code)
    if group and not profiles:
        logger.error("news-watch: 未找到监控组 %s", group)
        return 1
    if not profiles:
        logger.warning("news-watch: 无可用 profile，skip")
        return 0

    pushed = 0
    for profile in profiles:
        with ApplyStockProfile(profile):
            code = config.STOCK_CODE
            name = config.STOCK_NAME
            logger.info("news-watch: 扫描 %s(%s)...", name, code)
            try:
                data = FetchAllData()
                news_bundle = FetchNews(data)
                news_bundle = AnalyzeNewsImpact(news_bundle, data)
                verdict = EvaluateNewsAlert(
                    news_bundle,
                    float(news_bundle.get("news_score", 0) or 0),
                )
                if not verdict.get("worthy"):
                    logger.info(
                        "news-watch: skip %s — %s",
                        code,
                        verdict.get("reason", ""),
                    )
                    continue
                headline = str(verdict.get("headline") or "").strip()
                if not headline:
                    items = news_bundle.get("items") or []
                    headline = (
                        str(items[0].get("title", "")) if items else NEWS_WATCH_LABEL
                    )
                group_id = config.STOCK_GROUP or group or "default"
                if not TryClaimNewsAlert(headline, group=group_id):
                    continue
                news_bundle = EnrichCatalystWithLlm(
                    news_bundle, data, trigger_headline=headline,
                )
                logger.info(
                    "news-watch: 触发推送 %s — %s (%s)",
                    code,
                    headline[:60],
                    verdict.get("reason"),
                )
                _JobForCurrentProfile(
                    session_label=NEWS_WATCH_LABEL,
                    force_run=True,
                    prefetch={"data": data, "news_bundle": news_bundle},
                    trigger_headline=headline,
                )
                pushed += 1
            except Exception as exc:
                logger.exception("news-watch: %s(%s) 异常: %s", name, code, exc)
    logger.info("news-watch: 完成，推送 %d 只", pushed)
    return 0


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
    if not TryAcquireSchedulerLock():
        sys.exit(1)

    push_times = config.PUSH_TIMES
    logger.info("定时模式启动，每日推送时间点: %s", ", ".join(push_times))
    logger.info("手动: python main.py --now ｜ 持仓: python main.py --portfolio（已暂停）")
    logger.info(
        "若已配置 cron-job.org 触发 GitHub Actions，请勿同时运行本地定时调度，"
        "否则同一时刻会重复推送。"
    )

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

    if args.news_watch:
        logger.info(
            "股票分析工具 — 模式:资讯哨兵 group=%s ignore_window=%s",
            args.group or "全部",
            args.ignore_watch_window,
        )
        code = JobNewsWatch(
            group=args.group,
            stock_code=args.stock_code,
            ignore_window=args.ignore_watch_window,
        )
        if code:
            sys.exit(code)
        return

    if args.scheduled:
        label = GetSessionLabel(args.push_time)
        profiles = FilterProfilesByGroup(config.ResolveStockProfiles(), args.group)
        profiles = FilterProfilesByCode(profiles, args.stock_code)
        logger.info(
            "股票分析工具 — 模式:外部调度 push_time=%s 时段=%s group=%s 监控%d只股票",
            args.push_time or "auto",
            label,
            args.group or "全部",
            len(profiles),
        )
        Job(
            session_label=label,
            push_time=args.push_time,
            force_run=False,
            stock_code=args.stock_code,
            group=args.group,
        )
        return

    is_manual = args.now or config.MANUAL_RUN
    profiles = FilterProfilesByGroup(config.ResolveStockProfiles(), args.group)
    profiles = FilterProfilesByCode(profiles, args.stock_code)
    logger.info(
        "股票分析钉钉推送工具启动 — 模式:%s group=%s 监控%d只股票",
        "手动日常" if is_manual else "定时",
        args.group or "全部",
        len(profiles),
    )

    if is_manual:
        Job(force_run=True, stock_code=args.stock_code, group=args.group)
        return

    if args.local_scheduler or config.ENABLE_LOCAL_SCHEDULER:
        RunScheduler()
        return

    logger.info(
        "本地定时调度已关闭（方案 A：由 cron-job.org → GitHub Actions 推送）。"
        "手动推送: python main.py --now ｜ "
        "分组推送: python main.py --now --group oil_short ｜ "
        "单股推送: python main.py --now --stock-code 301201 ｜ "
        "如需本地常驻调度: python main.py --local-scheduler 或 .env 设 ENABLE_LOCAL_SCHEDULER=true"
        "｜ 资讯哨兵: python main.py --news-watch --group pharma"
    )


if __name__ == "__main__":
    try:
        Main()
    except KeyboardInterrupt:
        logger.info("程序已停止")
        sys.exit(0)
