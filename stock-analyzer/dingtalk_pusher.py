"""钉钉推送模块 — 加签 + Markdown 报告 + 发送。"""

from __future__ import annotations

import base64
import hashlib
import hmac
import re
import time
import urllib.parse
from typing import Any

import requests

import config
from report_web import PublishDetailReport
from trade_zones import ResolveTierRecommended
from utils import (
    FormatAmount,
    FormatPercent,
    FormatPrice,
    FormatVolume,
    NowStr,
    SetupLogger,
)

logger = SetupLogger(config.LOG_LEVEL)


def BuildSignedUrl(webhook: str, secret: str) -> str:
    """生成带加签参数的 Webhook URL。"""
    if not secret:
        return webhook
    timestamp = str(round(time.time() * 1000))
    string_to_sign = f"{timestamp}\n{secret}"
    hmac_code = hmac.new(
        secret.encode("utf-8"),
        string_to_sign.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).digest()
    sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
    separator = "&" if "?" in webhook else "?"
    return f"{webhook}{separator}timestamp={timestamp}&sign={sign}"


def ConfidenceStars(level: int) -> str:
    """将信心指数转为星级字符串。"""
    filled = min(level, 5)
    empty = 5 - filled
    return "⭐" * filled + "☆" * empty


def _SectionDivider() -> str:
    """章节分隔线。"""
    return "---"


def _BoldPrice(value: float, accent: str = "") -> str:
    """价格加粗，可选涨跌色。"""
    text = FormatPrice(value)
    if accent == "up":
        return f'<font color="#E53935">**{text}**</font>'
    if accent == "down":
        return f'<font color="#43A047">**{text}**</font>'
    return f"**{text}**"


def _BoldPriceRange(low: float, high: float) -> str:
    """价格区间加粗。"""
    return f"**{FormatPrice(low)} ~ {FormatPrice(high)}**"


def _EmPercentColored(value: float) -> str:
    """百分比加粗着色（红涨绿跌）。"""
    text = FormatPercent(value)
    if value > 0:
        return f'<font color="#E53935">**{text}**</font>'
    if value < 0:
        return f'<font color="#43A047">**{text}**</font>'
    return f"**{text}**"


def _EmScore(score: float, decimal: int = 0) -> str:
    """评分数值加粗着色。"""
    text = f"{score:+.{decimal}f}" if decimal > 0 else f"{score:+.0f}"
    if score > 0:
        return f'<font color="#E53935">**{text}**</font>'
    if score < 0:
        return f'<font color="#43A047">**{text}**</font>'
    return f"**{text}**"


def _EmAction(action: str) -> str:
    """操作建议加粗，买卖方向着色。"""
    text = action.strip()
    if not text:
        return "**—**"
    if any(k in text for k in ("卖", "减", "清")):
        return f'<font color="#43A047">**{text}**</font>'
    if any(k in text for k in ("买", "加", "入")):
        return f'<font color="#E53935">**{text}**</font>'
    return f"**{text}**"


def _HighlightMetricsInText(text: str) -> str:
    """依据/信号文本中的关键数值加粗。"""
    if not text or "**" in text:
        return text
    result = str(text)
    result = re.sub(
        r"(\d+\.\d{1,2})\s*[~～-]\s*(\d+\.\d{1,2})",
        r"**\1 ~ \2**",
        result,
    )
    result = re.sub(
        r"(MA\d+=)(\d+\.?\d*)",
        r"\1**\2**",
        result,
    )
    result = re.sub(
        r"(RSI\(\d+\)=)(\d+\.?\d*)",
        r"\1**\2**",
        result,
    )
    result = re.sub(
        r"(量比=)(\d+\.?\d*)",
        r"\1**\2**",
        result,
    )
    result = re.sub(
        r"(?<!\*\*)(?<![\d.])(\d+\.\d{1,2})(?!\*\*)(?![\d.])",
        r"**\1**",
        result,
    )
    result = re.sub(r"(\d+)(天|万元|条|股)", r"**\1**\2", result)
    result = re.sub(r"([+-]?\d+\.?\d*%)", r"**\1**", result)
    return result


def _HighlightNewsSignalTitle(text: str) -> str:
    """关键资讯前缀（个股利空/板块利好等）加粗着色。"""
    match = re.match(r"^((?:个股|板块|政策|宏观)(利好|利空))(：)", text)
    if not match:
        return _HighlightMetricsInText(text)
    prefix = match.group(1)
    direction = match.group(2)
    body = text[match.end():]
    if direction == "利好":
        tag = f'<font color="#E53935">**{prefix}：**</font>'
    else:
        tag = f'<font color="#43A047">**{prefix}：**</font>'
    return f"{tag}{_HighlightMetricsInText(body)}"


def _SplitRuleItems(rules: list[Any] | None) -> list[str]:
    """将规则列表拆为逐条可读项。"""
    if not rules:
        return []
    items: list[str] = []
    for raw in rules:
        text = str(raw).strip()
        if not text:
            continue
        for part in re.split(r"[；;]", text):
            part = part.strip()
            if part:
                items.append(part)
    return items


def _BuildNumberedBulletItems(items: list[Any], empty_text: str = "") -> list[str]:
    """构建编号 bullet 列表（适配钉钉换行）。"""
    lines: list[str] = []
    numbered = 0
    for raw in items:
        text = str(raw).strip()
        if not text:
            continue
        numbered += 1
        lines.append(f"- **{numbered}.** {_HighlightMetricsInText(text)}")
    if not lines and empty_text:
        lines.append(f"- {empty_text}")
    return lines


def _BuildPlainBulletItems(items: list[Any], empty_text: str = "") -> list[str]:
    """构建无编号 bullet 列表（适配钉钉换行）。"""
    lines: list[str] = []
    for raw in items:
        text = str(raw).strip()
        if text:
            lines.append(f"- {_HighlightMetricsInText(text)}")
    if not lines and empty_text:
        lines.append(f"- {empty_text}")
    return lines


def _BuildDetailedReasonsSection(advice: dict[str, Any]) -> list[str]:
    """构建详细依据章节。"""
    lines: list[str] = [
        "",
        "**买入依据**",
        "",
    ]
    lines.extend(_BuildNumberedBulletItems(
        advice.get("buy_reasons", []),
        "暂无明确买入信号",
    ))
    lines.extend(["", "**卖出依据**", ""])
    lines.extend(_BuildNumberedBulletItems(
        advice.get("sell_reasons", []),
        "暂无明确卖出信号",
    ))
    lines.append("")
    return lines


def _BuildSignalListSection(signals: list[Any], empty_text: str) -> list[str]:
    """构建信号列表章节。"""
    lines = _BuildPlainBulletItems(signals, empty_text)
    lines.append("")
    return lines


def _FormatChangeDisplay(change_pct: float, change_amt: float) -> str:
    """格式化涨跌幅（A股惯例：红涨绿跌）。"""
    sign = "+" if change_pct >= 0 else ""
    text = f"{sign}{change_pct:.2f}%（{sign}{change_amt:.2f}）"
    if change_pct > 0:
        return f'<font color="#E53935">**{text}**</font>'
    if change_pct < 0:
        return f'<font color="#43A047">**{text}**</font>'
    return f"**{text}**"


def _DirectionBadge(direction: str, icon: str) -> str:
    """方向预测徽章。"""
    if "涨" in direction:
        return f'<font color="#E53935">**{direction} {icon}**</font>'
    if "跌" in direction:
        return f'<font color="#43A047">**{direction} {icon}**</font>'
    return f"**{direction} {icon}**"


def _ScoreBar(label: str, score: float, weight_pct: int) -> str:
    """单行维度评分。"""
    return (
        f"- **{label}** {_EmScore(score)}  "
        f"<font color=\"#9E9E9E\">（{weight_pct}%）</font>"
    )


def _GetDirectionIcon(direction: str) -> str:
    """方向对应图标。"""
    if "涨" in direction and "跌" not in direction:
        return "⬆️" if direction == "看涨" else "↗️"
    if "跌" in direction:
        return "⬇️" if direction == "看跌" else "↘️"
    return "➡️"


def _BuildPredictionTierBlock(
    tier: dict[str, Any],
    show_range: bool = False,
) -> list[str]:
    """构建单一层级预测块（列表分行，适配钉钉换行）。"""
    label = str(tier.get("label", ""))
    direction = str(tier.get("direction", "震荡"))
    icon = _GetDirectionIcon(direction)
    lines: list[str] = ["", f"**{label}**"]
    lines.append(f"- **方向** {_DirectionBadge(direction, icon)}")
    if show_range:
        change_low = float(tier.get("change_low", 0))
        change_high = float(tier.get("change_high", 0))
        lines.append(
            f"- **区间** **{FormatPercent(change_low)} ~ {FormatPercent(change_high)}**"
        )
    prediction = str(tier.get("prediction", "")).strip()
    if prediction:
        lines.append(f"- **研判** {_HighlightMetricsInText(prediction)}")
    advice = str(tier.get("advice", "")).strip()
    if advice:
        lines.append(f"- **操作建议** **{advice}**")
    lines.append("")
    return lines


def _BuildComprehensivePredictionSection(
    analysis: dict[str, Any],
    advice: dict[str, Any],
    ai_tag: str,
    detail_url: str | None = None,
    detail_local_hint: str = "",
) -> list[str]:
    """构建 AI 分层综合预测章节。"""
    predictions = advice.get("predictions") or {}
    horizon = advice.get("prediction_horizon") or {}
    weighted = float(analysis.get("weighted_score", 0))
    tech_score = float(analysis.get("tech_score", 0))
    fund_score = float(analysis.get("fund_score", 0))
    sector_score = float(analysis.get("sector_score", 0))
    news_score = float(analysis.get("news_score", 0))
    news_signals = list(analysis.get("news_signals", []))

    near = predictions.get("near_term") or {}
    short_t = predictions.get("short_term") or {}
    long_t = predictions.get("long_term") or {}

    if not near and not short_t and not long_t:
        direction = analysis.get("direction", "震荡")
        icon = analysis.get("direction_icon", "➡️")
        return [
            "",
            f"- **方向** {_DirectionBadge(direction, icon)}",
            f"- **信心** {ConfidenceStars(analysis.get('confidence', 1))}",
            f"- **量化综合** {_EmScore(weighted, 1)}",
            f"- **预测区间** **{FormatPercent(float(analysis.get('change_low', 0)))} ~ "
            f"{FormatPercent(float(analysis.get('change_high', 0)))}**",
            "",
        ]

    lines: list[str] = [
        "",
        f"- **预测Horizon** {horizon.get('near_term_label', '近端预测')}"
        f"（{horizon.get('session_label', '')}）",
    ]
    near_target = str(horizon.get("near_term_target", "")).strip()
    if near_target:
        lines.append(f"- **预测对象** {near_target}")
    lines.append(f"- **量化综合** {_EmScore(weighted, 1)}")
    if near:
        lines.extend(_BuildPredictionTierBlock(near, show_range=True))
    if short_t:
        lines.extend(_BuildPredictionTierBlock(short_t))
    if long_t:
        lines.extend(_BuildPredictionTierBlock(long_t))

    news_brief = _BuildNewsBriefSection(advice, ai_tag)
    if news_brief:
        lines.extend(news_brief)

    detail_link = _BuildDetailReportLinkSection(detail_url, detail_local_hint)
    if detail_link:
        lines.extend(detail_link)

    lines.extend([
        "",
        "**四维评分（量化参考）**",
        _ScoreBar("技术", tech_score, 35),
        _ScoreBar("资金", fund_score, 30),
        _ScoreBar("板块", sector_score, 20),
        _ScoreBar("资讯", news_score, 15),
        "",
    ])
    if news_signals:
        lines.extend(["**关键资讯**"])
        for signal in news_signals[:3]:
            text = str(signal).strip()
            if text:
                lines.append(f"- {_HighlightNewsSignalTitle(text)}")
        lines.append("")
    return lines


def _BuildNewsBriefSection(
    advice: dict[str, Any],
    ai_tag: str,
) -> list[str]:
    """构建钉钉精简版资讯综合块（不含四维长列表）。"""
    news_summary = str(advice.get("news_summary", "")).strip()
    sector_impact = str(advice.get("sector_impact", "")).strip()
    policy_impact = str(advice.get("policy_impact", "")).strip()
    news_impact = str(advice.get("news_impact", "")).strip()
    news_conclusion = str(advice.get("news_conclusion", "")).strip()

    news_bundle = advice.get("news_bundle") or {}
    relevant_items: list[dict[str, Any]] = list(
        news_bundle.get("relevant_items")
        or news_bundle.get("scored_items", [])
    )
    bullish = [i for i in relevant_items if i.get("impact") == "涨"]
    bearish = [i for i in relevant_items if i.get("impact") == "跌"]

    horizon = advice.get("prediction_horizon") or {}
    is_trading_day = bool(horizon.get("is_trading_day", True))
    price_move = advice.get("price_move") or {}
    move_headline = ""
    if not is_trading_day:
        move_headline = str(price_move.get("headline", "")).strip()
    elif abs(float(price_move.get("move_pct", 0) or 0)) >= 2:
        move_headline = str(price_move.get("headline", "")).strip()

    has_content = bool(
        news_summary or sector_impact or policy_impact
        or news_impact or news_conclusion or bullish or bearish or move_headline
    )
    if not has_content:
        return []

    lines: list[str] = ["", f"**资讯综合{ai_tag}**"]
    if news_summary:
        lines.append(f"- {_TruncateText(news_summary, 120)}")
    if policy_impact:
        lines.append(f"- **政策** {_TruncateText(policy_impact, 120)}")
    if sector_impact:
        lines.append(f"- **板块** {_TruncateText(sector_impact, 80)}")
    conclusion = news_impact or news_conclusion
    if conclusion:
        lines.append(f"- **方向** {_TruncateText(conclusion, 100)}")
    if bullish:
        item = bullish[0]
        reason = str(item.get("impact_reason") or item.get("title", "")).strip()
        if reason:
            lines.append(f"- **利好** {_TruncateText(reason, 80)}")
    if bearish:
        item = bearish[0]
        reason = str(item.get("impact_reason") or item.get("title", "")).strip()
        if reason:
            lines.append(f"- **利空** {_TruncateText(reason, 80)}")
    if move_headline:
        label = "上日涨跌" if not is_trading_day else "涨跌主因"
        lines.append(f"- **{label}** {_TruncateText(move_headline, 100)}")
    lines.append("")
    return lines


def _BuildDetailReportLinkSection(
    detail_url: str | None = None,
    detail_local_hint: str = "",
) -> list[str]:
    """构建醒目完整深度报告链接（置于资讯综合之后）。"""
    if detail_url:
        return [
            "",
            _SectionDivider(),
            "",
            '<font color="#1677FF">**📄 完整深度报告**</font>',
            "",
            f'<font color="#1677FF">**[👉 点击查看：涨跌归因 · 详细依据 · 资讯解读]({detail_url})**</font>',
            "",
            "<font color=\"#757575\">五维涨跌归因 / 买卖详细依据 / 四维资讯解读</font>",
            "",
        ]
    if detail_local_hint:
        return [
            "",
            _SectionDivider(),
            "",
            "**📄 完整深度报告**",
            f"- {detail_local_hint}",
            "",
        ]
    return []


def _NextSection(section: list[int], title: str) -> str:
    """生成带中文序号的章节标题。"""
    cn_nums = ["", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]
    idx = section[0]
    section[0] += 1
    num = cn_nums[idx] if idx < len(cn_nums) else str(idx)
    return f"## {num}、{title}"


def _TruncateText(text: str, max_len: int) -> str:
    """截断过长文本。"""
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _FormatNewsTime(time_str: str) -> str:
    """将资讯时间格式化为简短可读形式。"""
    if not time_str:
        return ""
    text = time_str.strip()
    if len(text) >= 16 and text[4] == "-":
        return text[5:16]
    if len(text) == 8 and text.isdigit():
        return f"{text[4:6]}-{text[6:8]}"
    return _TruncateText(text, 11)


def _ImpactBadge(impact: str) -> str:
    """涨跌影响徽章（A股惯例：红涨绿跌）。"""
    if impact == "涨":
        return '<font color="#E53935">**▲ 涨**</font>'
    if impact == "跌":
        return '<font color="#43A047">**▼ 跌**</font>'
    return '<font color="#9E9E9E">— 中性</font>'


def _ImpactSectionTitle(title: str, impact: str) -> str:
    """重点利好/利空分节标题。"""
    if impact == "涨":
        return f'<font color="#E53935">**▲ {title}**</font>'
    if impact == "跌":
        return f'<font color="#43A047">**▼ {title}**</font>'
    return f"**{title}**"


def _HasDirectPriceImpact(item: dict[str, Any]) -> bool:
    """判断资讯是否对股价有明确涨跌影响。"""
    return str(item.get("impact", "")) in ("涨", "跌")


_NO_DIRECT_IMPACT_COPY: dict[str, str] = {
    "sector": "暂无直接影响该股价涨跌的板块/行业事件，可作背景参考",
    "policy": "暂无直接影响该股价涨跌的政策/监管事件，可作背景参考",
    "macro": "暂无直接影响该股价涨跌的宏观事件，可作背景参考",
    "web_search": "联网搜索暂无直接影响涨跌的结果，可作背景参考",
}


def _NormalizeTitleKey(title: str) -> str:
    """归一化标题用于去重。"""
    text = re.sub(r"[（()）\\s\"\"''、，。：:]", "", title)
    text = re.sub(r"\d+\.?\d*", "N", text)
    return text[:40]


def _DedupeNewsItems(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按标题相似度去重，保留 strength 最高的一条。"""
    sorted_items = sorted(
        items,
        key=lambda x: (-int(x.get("strength", 0) or 0), str(x.get("time", ""))),
        reverse=False,
    )
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for item in sorted_items:
        key = _NormalizeTitleKey(str(item.get("title", "")))
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _ContentMaxLen(category: str) -> int:
    """按类别返回摘要长度上限。"""
    if category == "policy":
        return 150
    if category == "macro":
        return 80
    return 70


def _TitleMaxLen(category: str) -> int:
    """按类别返回标题长度上限。"""
    return 60 if category == "policy" else 48


def _FormatNewsItemBlock(
    item: dict[str, Any],
    index: int,
) -> list[str]:
    """格式化单条资讯（统一卡片样式）。"""
    cat = str(item.get("category", "stock"))
    time_tag = _FormatNewsTime(str(item.get("time", "")))
    title = _TruncateText(str(item.get("title", "")), _TitleMaxLen(cat))
    source = str(item.get("source", "")).strip()
    impact = str(item.get("impact", "中性"))
    impact_badge = _ImpactBadge(impact)
    time_part = f"`{time_tag}` " if time_tag else ""
    source_part = f"（{source}）" if source else ""
    lines = [
        f"- **{index}.** {impact_badge}  {time_part}**{title}**{source_part}",
    ]

    reason = str(item.get("impact_reason", "")).strip()
    if reason:
        lines.append(f"  - **影响** {reason}")

    content = str(item.get("content", "")).strip()
    if content:
        lines.append(f"  - **摘要** {_TruncateText(content, _ContentMaxLen(cat))}")

    return lines


def _BuildNewsGroupSection(
    title: str,
    items: list[dict[str, Any]],
    limit: int,
    impact: str = "",
) -> list[str]:
    """构建某一影响方向的分组资讯。"""
    if not items:
        return []
    deduped = _DedupeNewsItems(items)[:limit]
    section_title = _ImpactSectionTitle(title, impact) if impact else f"**{title}**"
    lines = [f"{section_title}（{len(deduped)}条）", ""]
    for i, item in enumerate(deduped, 1):
        lines.extend(_FormatNewsItemBlock(item, i))
        lines.append("")
    return lines


def _BuildNewsByCategorySections(
    relevant_items: list[dict[str, Any]],
) -> list[str]:
    """按四维分类构建资讯分节。"""
    category_meta: list[tuple[str, str, int]] = [
        ("stock", "【个股资讯】", 5),
        ("sector", "【板块/行业】", 4),
        ("policy", "【政策/监管】", 6),
        ("web_search", "【联网搜索】", 4),
        ("macro", "【宏观事件】", 3),
    ]
    lines: list[str] = []
    for cat_key, section_title, limit in category_meta:
        cat_items = [
            i for i in relevant_items
            if str(i.get("category", "stock")) == cat_key
        ]
        if not cat_items:
            continue
        deduped = _DedupeNewsItems(cat_items)
        impactful = [i for i in deduped if _HasDirectPriceImpact(i)]

        if cat_key == "stock":
            display_items = deduped[:limit]
            lines.append(f"**{section_title}**（{len(display_items)}条）")
            lines.append("")
            for i, item in enumerate(display_items, 1):
                lines.extend(_FormatNewsItemBlock(item, i))
                lines.append("")
            continue

        if impactful:
            display_items = impactful[:limit]
            lines.append(f"**{section_title}**（{len(display_items)}条有影响）")
            lines.append("")
            for i, item in enumerate(display_items, 1):
                lines.extend(_FormatNewsItemBlock(item, i))
                lines.append("")
            continue

        no_impact_copy = _NO_DIRECT_IMPACT_COPY.get(cat_key, "暂无直接影响该股价涨跌的事件")
        lines.append(f"**{section_title}**")
        lines.append(f"- {no_impact_copy}（共采集 {len(deduped)} 条）")
        lines.append("")
    return lines


def _BuildKeyImpactHighlights(
    relevant_items: list[dict[str, Any]],
    limit: int = 3,
) -> list[str]:
    """构建跨维度重点利好/利空精选。"""
    bullish = sorted(
        [i for i in relevant_items if i.get("impact") == "涨"],
        key=lambda x: x.get("strength", 0),
        reverse=True,
    )
    bearish = sorted(
        [i for i in relevant_items if i.get("impact") == "跌"],
        key=lambda x: x.get("strength", 0),
        reverse=True,
    )
    lines: list[str] = []
    if bullish:
        lines.extend(_BuildNewsGroupSection("重点利好", bullish, limit=limit, impact="涨"))
    if bearish:
        lines.extend(_BuildNewsGroupSection("重点利空", bearish, limit=limit, impact="跌"))
    return lines


def _BuildNewsInterpretationSection(
    advice: dict[str, Any],
    ai_tag: str,
) -> list[str]:
    """构建资讯解读章节（四维分类 + 重点利好/利空）。"""
    news_bundle = advice.get("news_bundle") or {}
    news_summary = advice.get("news_summary", "")
    sector_impact = advice.get("sector_impact", "")
    policy_impact = advice.get("policy_impact", "")
    news_impact = advice.get("news_impact", "")

    relevant_items: list[dict[str, Any]] = list(
        news_bundle.get("relevant_items")
        or news_bundle.get("scored_items", [])
    )

    if not (relevant_items or news_summary or sector_impact or policy_impact):
        return []

    lines: list[str] = []

    stats = news_bundle.get("impact_stats", {})
    bullish = [i for i in relevant_items if i.get("impact") == "涨"]
    bearish = [i for i in relevant_items if i.get("impact") == "跌"]
    neutral = [i for i in relevant_items if i.get("impact") == "中性"]

    if news_summary or sector_impact or policy_impact or news_impact:
        lines.extend([f"**AI 研判{ai_tag}**", ""])
        if news_summary:
            lines.append(f"- {news_summary[:200]}")
        impacts: list[tuple[str, str]] = []
        if sector_impact:
            impacts.append(("板块", sector_impact))
        if policy_impact:
            impacts.append(("政策", policy_impact))
        if news_impact:
            impacts.append(("方向", news_impact))
        for label, content in impacts:
            max_len = 200 if label == "政策" else 120
            lines.append(f"- **{label}** {_TruncateText(content, max_len)}")

    by_category = stats.get("by_category", {})
    if stats:
        lines.extend([
            "",
            "**采集统计**",
            "",
            f"- **总量** 共采集 **{stats.get('total', 0)}** 条",
            f"- **分类** 个股 **{by_category.get('stock', 0)}**  "
            f"板块 **{by_category.get('sector', 0)}**  "
            f"政策 **{by_category.get('policy', 0)}**  "
            f"宏观 **{by_category.get('macro', 0)}**",
            f"- **影响** "
            f'<font color="#E53935">▲{len(bullish)}</font>  '
            f'<font color="#43A047">▼{len(bearish)}</font>  '
            f"已过滤无关 **{stats.get('irrelevant', 0)}** 条",
        ])

    category_lines = _BuildNewsByCategorySections(relevant_items)
    if category_lines:
        lines.extend(["", *category_lines])
    elif not bullish and not bearish and not neutral:
        lines.extend(["**📰 资讯概览**", "- 今日暂无显著相关资讯", ""])

    lines.extend(_BuildKeyImpactHighlights(relevant_items, limit=3))

    news_conclusion = str(advice.get("news_conclusion", "")).strip()
    if not news_conclusion:
        parts: list[str] = []
        direction_pred = str(advice.get("direction_prediction", "")).strip()
        news_impact_text = str(advice.get("news_impact", "")).strip()
        if direction_pred:
            parts.append(direction_pred)
        if news_impact_text and news_impact_text not in parts:
            parts.append(news_impact_text)
        if parts:
            news_conclusion = " ".join(parts)
        elif bullish or bearish:
            if len(bearish) > len(bullish):
                news_conclusion = (
                    f"资讯面偏空（利好{len(bullish)}条/利空{len(bearish)}条），"
                    "建议谨慎观望，不宜追涨。"
                )
            elif len(bullish) > len(bearish):
                news_conclusion = (
                    f"资讯面偏多（利好{len(bullish)}条/利空{len(bearish)}条），"
                    "可适度关注，但仍需结合技术面确认。"
                )
            else:
                news_conclusion = "资讯面多空交织，建议以技术面和资金面为主导，资讯作辅助参考。"
        elif neutral:
            news_conclusion = (
                f"当前 {len(neutral)} 条资讯整体偏中性，"
                "对股价无明确方向指引，建议以技术面和资金面为主要决策依据。"
            )

    if news_conclusion:
        lines.extend([
            "",
            f"**综合建议{ai_tag}**",
            f"- {_HighlightMetricsInText(news_conclusion)}",
            "",
        ])

    return lines


def _BuildPricingNote(advice: dict[str, Any]) -> str:
    """生成定价说明文案。"""
    rationale = str(advice.get("pricing_rationale", "")).strip()
    source = str(advice.get("pricing_source", ""))
    if rationale:
        prefix = "AI 综合定价" if source in ("ai", "ai_repaired") else "定价说明"
        return f"{prefix}：{rationale}"
    if source == "ai":
        return "买卖区间由 AI 综合 K 线支撑/压力、资讯与四维评分给出"
    if source == "ai_repaired":
        return "买卖区间基于 AI 分析，已对不满足约束的价位做智能修正"
    if advice.get("llm_used"):
        return "AI 定价未能完全满足约束，本次回退 K 线量化支撑/压力位"
    return "买卖区间参考 K 线量化支撑/压力位，并结合技术面与资讯研判"


def ResolveTradeDisplayLevels(advice: dict[str, Any]) -> dict[str, Any]:
    """解析与分档区间一致的展示用支撑/压力位。"""
    zones = advice.get("trade_zones") or {}
    at1 = zones.get("add_tier1", {})
    at2 = zones.get("add_tier2", {})
    st1 = zones.get("sell_tier1", {})
    st2 = zones.get("sell_tier2", {})

    buy_low = float(advice.get("buy_low", at1.get("low", 0)) or 0)
    buy_high = float(advice.get("buy_high", at2.get("high", 0)) or 0)
    sell_low = float(advice.get("sell_low", st1.get("low", 0)) or 0)
    sell_high = float(advice.get("sell_high", st2.get("high", 0)) or 0)

    if buy_low > buy_high and buy_high > 0:
        buy_low, buy_high = buy_high, buy_low
    if sell_low > sell_high and sell_high > 0:
        sell_low, sell_high = sell_high, sell_low

    display_s1 = buy_high
    display_s2 = buy_low
    display_r1 = sell_low
    display_r2 = sell_high

    stop_loss = float(advice.get("stop_loss", 0) or 0)
    if display_s2 > 0 and (stop_loss <= 0 or stop_loss >= display_s2):
        stop_loss = round(display_s2 * 0.98, 2)

    rule_s1 = float(advice.get("rule_s1", advice.get("s1", 0)) or 0)
    rule_s2 = float(advice.get("rule_s2", advice.get("s2", 0)) or 0)
    rule_r1 = float(advice.get("rule_r1", advice.get("r1", 0)) or 0)
    rule_r2 = float(advice.get("rule_r2", advice.get("r2", 0)) or 0)

    def _Differs(a: float, b: float) -> bool:
        return a > 0 and b > 0 and abs(a - b) > 0.01

    kline_differs = (
        _Differs(rule_s1, display_s1)
        or _Differs(rule_s2, display_s2)
        or _Differs(rule_r1, display_r1)
        or _Differs(rule_r2, display_r2)
    )

    return {
        "buy_low": buy_low,
        "buy_high": buy_high,
        "sell_low": sell_low,
        "sell_high": sell_high,
        "s1": display_s1,
        "s2": display_s2,
        "r1": display_r1,
        "r2": display_r2,
        "stop_loss": stop_loss,
        "kline_differs": kline_differs,
        "rule_s1": rule_s1,
        "rule_s2": rule_s2,
        "rule_r1": rule_r1,
        "rule_r2": rule_r2,
        "trade_zones": zones,
    }


def _FormatTierBlock(
    label: str,
    tier: dict[str, Any],
    show_stop: bool = False,
    side: str = "buy",
) -> list[str]:
    """格式化单档区间块（推荐价 + 参考区间，适配钉钉换行）。"""
    low_val = float(tier.get("low", 0) or 0)
    high_val = float(tier.get("high", 0) or 0)
    tier_side: str = "sell" if side == "sell" else "buy"
    rec = ResolveTierRecommended(tier, tier_side)  # type: ignore[arg-type]
    span = high_val - low_val if high_val > low_val else 0.0
    width_pct = span / rec * 100 if rec > 0 else 0.0
    source = str(tier.get("recommended_source", "")).strip()
    source_tag = "（AI）" if source == "ai" else "（技术锚点）"
    accent = "up" if tier_side == "buy" else "down"

    lines: list[str] = ["", f"**{label}**"]
    if rec > 0:
        lines.append(f"- **推荐价位{source_tag}** {_BoldPrice(rec, accent)}")
    if span > 0.01 and (span > 0.6 or width_pct > 2.0):
        lines.append(
            f"- **参考区间** {_BoldPriceRange(low_val, high_val)}（可接受波动）"
        )
    elif span > 0.01:
        lines.append(f"- **价格区间** {_BoldPriceRange(low_val, high_val)}")
    logic = str(tier.get("logic", tier.get("note", ""))).strip()
    if logic:
        lines.append(f"- **逻辑** {_HighlightMetricsInText(logic)}")
    rules = _SplitRuleItems(tier.get("rules"))
    for i, rule in enumerate(rules, 1):
        lines.append(f"- **规则{i}** {_HighlightMetricsInText(rule)}")
    if show_stop:
        stop = float(tier.get("stop_loss", 0) or 0)
        if stop > 0:
            lines.append(
                f"- **止损** 跌破 {_BoldPrice(stop, 'down')} 卖出加仓部分"
            )
    lines.append("")
    return lines


def _FormatNoAddZones(zones: list[dict[str, Any]]) -> list[str]:
    """格式化不宜加仓区间。"""
    if not zones:
        return []
    lines: list[str] = ["", "**不宜加仓**"]
    for z in zones:
        lo = float(z.get("low", 0) or 0)
        hi = float(z.get("high", 0) or 0)
        reason = str(z.get("reason", "")).strip()
        if hi >= 9999:
            price_text = f"{_BoldPrice(lo, 'up')} 以上"
        elif lo <= 0:
            price_text = f"有效跌破 {_BoldPrice(hi, 'down')}"
        else:
            price_text = _BoldPriceRange(lo, hi)
        if reason:
            lines.append(f"- {price_text}：{reason}")
        else:
            lines.append(f"- {price_text}")
    lines.append("")
    return lines


def _BuildTradeAdviceSection(
    advice: dict[str, Any],
    ai_tag: str,
    buy_icon: str,
    sell_icon: str,
) -> list[str]:
    """构建买卖建议章节（豆包维度分档）。"""
    levels = ResolveTradeDisplayLevels(advice)
    plan = advice.get("trade_plan") or {}
    zones = plan if plan.get("add_tier1") else (advice.get("trade_zones") or {})
    at1 = zones.get("add_tier1", {})
    at2 = zones.get("add_tier2", {})
    st1 = zones.get("sell_tier1", {})
    st2 = zones.get("sell_tier2", {})
    buy_action = str(advice.get("buy_action", "观望")).strip() or "观望"
    sell_action = str(advice.get("sell_action", "持有")).strip() or "持有"
    buy_reasons = advice.get("buy_reasons", [])
    sell_reasons = advice.get("sell_reasons", [])

    lines: list[str] = []
    risk = str(plan.get("risk_warning", "")).strip()
    if risk:
        lines.extend(["", f"- **风险提示** {risk}"])

    lines.extend([
        "",
        "- 综合 K 线技术锚点、箱体/VWAP、四维评分及 AI 研判",
        "",
        f"**{buy_icon} 买入建议**",
        f"- **动作** {_EmAction(buy_action)}",
    ])
    lines.extend(_FormatTierBlock("第一加仓（优先等待）", at1, show_stop=True))
    lines.extend(_FormatTierBlock("第二加仓（次选）", at2, show_stop=False))
    lines.extend(_FormatNoAddZones(plan.get("no_add_zones", [])))
    if buy_reasons:
        lines.append(f"- **买入依据** {_HighlightMetricsInText(str(buy_reasons[0]))}")

    lines.extend([
        "",
        f"**{sell_icon} 卖出建议（做T降本）**",
        f"- **动作** {_EmAction(sell_action)}",
    ])
    lines.extend(_FormatTierBlock("第一卖出（先减）", st1, side="sell"))
    lines.extend(_FormatTierBlock("第二卖出（再减）", st2, side="sell"))
    if sell_reasons:
        lines.append(f"- **卖出依据** {_HighlightMetricsInText(str(sell_reasons[0]))}")

    discipline = plan.get("discipline") or []
    if discipline:
        lines.extend(["", "**加仓纪律**"])
        for i, d in enumerate(discipline, 1):
            lines.append(f"- **{i}.** {_HighlightMetricsInText(str(d))}")

    bd = plan.get("breakdown_plan") or {}
    if bd.get("level", 0) > 0:
        next_sup = bd.get("next_support", 0)
        next_text = (
            f"（下一档 {_BoldPrice(float(next_sup))}）" if next_sup > 0 else ""
        )
        lines.extend([
            "",
            f"- **跌破应对** 有效跌破 {_BoldPrice(float(bd.get('level', 0)), 'down')}："
            f"{bd.get('action', '')}{next_text}",
        ])

    rationale = str(advice.get("pricing_rationale", "")).strip()
    pricing_note = _BuildPricingNote(advice)
    lines.extend(["", f"- **定价说明** {pricing_note}"])
    if rationale and rationale not in pricing_note:
        lines.append(f"- {rationale}")

    if levels["kline_differs"]:
        lines.append(
            f"- **K线均线参考** S1={_BoldPrice(levels['rule_s1'])} "
            f"S2={_BoldPrice(levels['rule_s2'])} ｜ "
            f"R1={_BoldPrice(levels['rule_r1'], 'up')} "
            f"R2={_BoldPrice(levels['rule_r2'], 'up')}"
        )

    strategy_text = str(advice.get("strategy", "")).strip()
    if strategy_text:
        lines.extend([
            "",
            f"**操作策略{ai_tag}**",
            f"- {strategy_text}",
            "",
        ])

    return lines


def _BuildReportHeaderSection(
    price: float,
    change_pct: float,
    change_amt: float,
    session_label: str,
    header_direction: str,
    header_icon: str,
    buy_action: str,
    sell_action: str,
    add_tier1: dict[str, Any],
    near_advice: str,
    counts_hdr: dict[str, int],
    fetched_at_hdr: str,
    horizon: dict[str, Any] | None = None,
) -> list[str]:
    """构建报告顶部速览（大标题分行，适配钉钉 Markdown 换行）。"""
    lines: list[str] = [
        f"# 📊 {config.STOCK_NAME}",
        f"## {config.STOCK_CODE} 分析报告",
        "",
        f"**{session_label}** ｜ {NowStr()}",
        "",
        f"**现价** {_BoldPrice(price, 'up' if change_pct >= 0 else 'down')}  "
        f"{_FormatChangeDisplay(change_pct, change_amt)}",
        "",
        f"**方向** {_DirectionBadge(header_direction, header_icon)}",
    ]
    if horizon:
        target = str(horizon.get("near_term_target", "")).strip()
        if target:
            lines.extend(["", f"**预测对象** {target}"])
    lines.extend([
        "",
        f"**买卖建议** 买 {_EmAction(buy_action)} ｜ 卖 {_EmAction(sell_action)}",
    ])
    if add_tier1.get("low") and add_tier1.get("high"):
        rec1 = ResolveTierRecommended(add_tier1, "buy")
        if rec1 > 0:
            src = "AI" if add_tier1.get("recommended_source") == "ai" else "技术锚点"
            lines.extend([
                "",
                f"**优先加仓价** {_BoldPrice(rec1, 'up')}（{src}）",
            ])
        else:
            lines.extend([
                "",
                f"**优先加仓区** {_BoldPriceRange(float(add_tier1['low']), float(add_tier1['high']))}",
            ])
    if near_advice.strip():
        lines.extend(["", f"**近端建议** {near_advice.strip()}"])
    if counts_hdr:
        lines.extend([
            "",
            "**资讯统计** "
            f"个股 **{counts_hdr.get('stock', 0)}**  "
            f"板块 **{counts_hdr.get('sector', 0)}**  "
            f"政策 **{counts_hdr.get('policy', 0)}**  "
            f"宏观 **{counts_hdr.get('macro', 0)}**",
        ])
    if fetched_at_hdr:
        lines.extend(["", f"**资讯采集** {fetched_at_hdr}"])
    lines.append("")
    return lines


def _BuildRealtimeSection(
    rt: dict[str, Any],
    price: float,
    prev_close: float,
    change_pct: float,
    change_amt: float,
    is_trading_day: bool = True,
) -> list[str]:
    """构建实时行情章节（列表分行，避免钉钉合并引用块）。"""
    if not rt:
        return ["", "实时行情数据暂不可用", ""]

    lines: list[str] = [""]
    if not is_trading_day:
        lines.append("- **说明** 休市中，以下行情为上一交易日收盘参考")
    lines.extend([
        f"- **现价** {_BoldPrice(price, 'up' if change_pct >= 0 else 'down')}  "
        f"｜ 昨收 {_BoldPrice(prev_close)}  "
        f"｜ {_FormatChangeDisplay(change_pct, change_amt)}",
        f"- **开/高/低** {_BoldPrice(float(rt.get('open', 0)))} / "
        f"{_BoldPrice(float(rt.get('high', 0)), 'up')} / "
        f"{_BoldPrice(float(rt.get('low', 0)), 'down')}",
        f"- **量/额** **{FormatVolume(rt.get('volume', 0))}** / "
        f"**{FormatAmount(rt.get('amount', 0))}**",
        f"- **换手/量比** **{rt.get('turnover', 0):.2f}%** / "
        f"**{rt.get('volume_ratio', 0):.2f}**",
        f"- **PE/PB** **{rt.get('pe', 0):.1f}** / **{rt.get('pb', 0):.2f}**",
        "",
    ])
    return lines


_DIM_CN_NUMS = ["", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]


def _BuildPriceMoveSection(
    price_move: dict[str, Any] | None,
    ai_tag: str,
) -> list[str]:
    """构建涨跌归因拆解章节。"""
    if not price_move:
        return []

    direction = str(price_move.get("move_direction", "平"))
    move_pct = float(price_move.get("move_pct", 0) or 0)
    headline = str(price_move.get("headline", "")).strip()
    disclaimer = str(price_move.get("disclaimer", "")).strip()
    brief = bool(price_move.get("brief"))
    source_tag = ai_tag if price_move.get("source") == "llm" else ""

    lines: list[str] = [
        "",
        f"- **归因摘要{source_tag}** {_EmAction(headline) if headline else '—'}",
    ]
    if disclaimer:
        lines.append(f"- {disclaimer}")

    if brief:
        gaps = price_move.get("evidence_gaps") or []
        if gaps:
            lines.append(f"- **证据缺口** {'；'.join(str(g) for g in gaps[:3])}")
        lines.append("")
        return lines

    dimensions = price_move.get("dimensions") or []
    for idx, dim in enumerate(dimensions, 1):
        if not isinstance(dim, dict):
            continue
        dim_title = str(dim.get("title", "")).strip()
        num = _DIM_CN_NUMS[idx] if idx < len(_DIM_CN_NUMS) else str(idx)
        section_title = f"**{num}、{dim_title}**"
        if dim.get("id") == "risks":
            section_title = f'<font color="#E53935">{section_title}</font>'
        elif dim.get("id") == "catalyst":
            section_title = f'<font color="#E53935">{section_title}</font>' if direction == "涨" else (
                f'<font color="#43A047">{section_title}</font>' if direction == "跌" else section_title
            )
        lines.extend(["", section_title, ""])

        points = dim.get("points") or []
        if not points:
            lines.append("- 暂无足够证据支撑该维度归因")
            continue
        for pt in points:
            if not isinstance(pt, dict):
                continue
            subtitle = str(pt.get("subtitle", "")).strip()
            content = str(pt.get("content", "")).strip()
            evidence = str(pt.get("evidence", "")).strip()
            if not content:
                continue
            bullet = f"- **{subtitle}** {content}" if subtitle else f"- {content}"
            lines.append(_HighlightMetricsInText(bullet))
            if evidence and evidence not in content:
                lines.append(f"  - 依据：{evidence}")

        limitations = str(dim.get("limitations", "")).strip()
        if limitations:
            lines.append(f"- **限制** {limitations}")

    gaps = price_move.get("evidence_gaps") or []
    if gaps:
        lines.extend(["", "**证据缺口**", ""])
        for gap in gaps[:5]:
            lines.append(f"- {gap}")

    lines.append("")
    return lines


_DISCLAIMER_WEB = (
    "以上分析由程序自动生成，仅供参考，不构成投资建议。股市有风险，投资需谨慎。"
)


def BuildDetailSectionsMarkdown(
    data: dict[str, Any],
    analysis: dict[str, Any],
    advice: dict[str, Any],
    session_label: str = "盘中",
    report_mode: str = "daily",
) -> str:
    """构建网页详细报告 Markdown（涨跌归因 + 详细依据 + 资讯解读）。"""
    rt = data.get("realtime") or {}
    change_pct = rt.get("change_pct", 0)
    ai_tag = "（AI）" if advice.get("llm_used") else ""
    lines: list[str] = [
        f"> {NowStr()} ｜ {session_label}",
        "",
    ]

    price_move = advice.get("price_move")
    detail_idx = 1
    if price_move:
        move_pct_hdr = float(price_move.get("move_pct", change_pct) or change_pct)
        pct_text = FormatPercent(move_pct_hdr)
        cn = _DIM_CN_NUMS[detail_idx] if detail_idx < len(_DIM_CN_NUMS) else str(detail_idx)
        lines.extend([
            _SectionDivider(),
            "",
            f'<h2 id="section-catalyst">{cn}、涨跌归因拆解（{pct_text}）</h2>',
        ])
        lines.extend(_BuildPriceMoveSection(price_move, ai_tag))
        detail_idx += 1

    cn = _DIM_CN_NUMS[detail_idx] if detail_idx < len(_DIM_CN_NUMS) else str(detail_idx)
    lines.extend([
        "",
        _SectionDivider(),
        "",
        f'<h2 id="section-reasons">{cn}、详细依据{ai_tag}</h2>',
    ])
    lines.extend(_BuildDetailedReasonsSection(advice))
    detail_idx += 1

    news_section = _BuildNewsInterpretationSection(advice, ai_tag)
    if news_section:
        cn = _DIM_CN_NUMS[detail_idx] if detail_idx < len(_DIM_CN_NUMS) else str(detail_idx)
        lines.extend([
            "",
            _SectionDivider(),
            "",
            f'<h2 id="section-news">{cn}、资讯解读{ai_tag}</h2>',
        ])
        lines.extend(news_section)

    lines.extend([
        "",
        _SectionDivider(),
        f"> ⚠️ {_DISCLAIMER_WEB}",
    ])
    return "\n".join(lines)


def BuildDingTalkReportMarkdown(
    data: dict[str, Any],
    analysis: dict[str, Any],
    advice: dict[str, Any],
    session_label: str = "盘中",
    report_mode: str = "daily",
    detail_url: str | None = None,
    detail_local_hint: str = "",
) -> str:
    """构建钉钉精简版 Markdown（不含涨跌归因/详细依据/资讯解读）。"""
    mode = advice.get("report_mode", report_mode)
    rt = data.get("realtime") or {}
    price = rt.get("price", analysis.get("indicators", {}).get("price", 0))
    prev_close = rt.get("prev_close", 0)
    change_pct = rt.get("change_pct", 0)
    change_amt = rt.get("change_amt", 0)

    direction = analysis.get("direction", "震荡")
    buy_icon = "🟢" if advice.get("buy_ok") else "🟡"
    sell_icon = "🔴" if advice.get("sell_ok") else "🟡"
    ai_tag = "（AI）" if advice.get("llm_used") else ""
    sec = [1]

    buy_action_hdr = str(advice.get("buy_action", "观望"))
    sell_action_hdr = str(advice.get("sell_action", "持有"))
    predictions_hdr = advice.get("predictions") or {}
    near_hdr = predictions_hdr.get("near_term") or {}
    header_direction = str(near_hdr.get("direction") or direction)
    header_icon = _GetDirectionIcon(header_direction)
    plan_hdr = advice.get("trade_plan") or {}
    at1_hdr = (
        plan_hdr.get("add_tier1")
        or advice.get("trade_zones", {}).get("add_tier1")
        or {}
    )
    news_bundle_hdr = advice.get("news_bundle") or {}
    counts_hdr = news_bundle_hdr.get("category_counts", {})
    horizon_hdr = advice.get("prediction_horizon") or {}
    is_trading_day = bool(horizon_hdr.get("is_trading_day", True))

    lines: list[str] = _BuildReportHeaderSection(
        float(price),
        float(change_pct),
        float(change_amt),
        session_label,
        header_direction,
        header_icon,
        buy_action_hdr,
        sell_action_hdr,
        at1_hdr,
        str(near_hdr.get("advice", "")),
        counts_hdr,
        str(news_bundle_hdr.get("fetched_at", "")),
        horizon=horizon_hdr,
    )

    lines.extend([
        _SectionDivider(),
        "",
        _NextSection(sec, "实时行情"),
    ])
    lines.extend(_BuildRealtimeSection(
        rt, float(price), float(prev_close), float(change_pct), float(change_amt),
        is_trading_day=is_trading_day,
    ))

    portfolio = advice.get("portfolio")
    if portfolio and mode != "daily":
        lines.extend([
            "",
            _NextSection(sec, "持仓与回本"),
            f"- 总持仓：**{portfolio.get('total_shares', 0)} 股**",
            f"- 加权成本：**{FormatPrice(portfolio.get('weighted_cost', 0))}**",
            f"- 回本目标价：**{FormatPrice(portfolio.get('breakeven_price', 0))}**"
            f"（需涨 {portfolio.get('distance_to_breakeven_pct', 0):+.1f}%）",
            f"- 浮动盈亏：{portfolio.get('total_pnl', 0):+.2f} 元"
            f"（{portfolio.get('total_pnl_pct', 0):+.2f}%）",
        ])
        lots = portfolio.get("lots", [])
        for i, lot in enumerate(lots, 1):
            lines.append(
                f"  - 仓{i}：成本 {lot.get('cost', 0):.3f} × {lot.get('shares', 0)}股"
            )
        if advice.get("breakeven_analysis"):
            lines.append(f"- 回本分析{ai_tag}：{advice['breakeven_analysis']}")
        recovery = advice.get("recovery_steps", [])
        if recovery:
            lines.append("- 回本/减亏路径：")
            for i, step in enumerate(recovery, 1):
                lines.append(f"  {i}. {step}")

    lines.extend([
        "",
        _SectionDivider(),
        "",
        _NextSection(sec, f"综合预测{ai_tag}"),
        "",
    ])
    lines.extend(_BuildComprehensivePredictionSection(
        analysis, advice, ai_tag,
        detail_url=detail_url,
        detail_local_hint=detail_local_hint,
    ))

    lines.extend([
        "",
        _SectionDivider(),
        "",
        _NextSection(sec, f"买卖建议{ai_tag}"),
    ])
    lines.extend(_BuildTradeAdviceSection(advice, ai_tag, buy_icon, sell_icon))

    t_trade = advice.get("t_trade") or {}
    if t_trade.get("enabled") and mode != "daily":
        lines.extend([
            "",
            f"**做T建议{ai_tag}：**",
            f"- 低吸：{FormatPrice(t_trade.get('buy_price', 0))} ｜ "
            f"高抛：{FormatPrice(t_trade.get('sell_price', 0))} ｜ "
            f"参考股数：{t_trade.get('shares', 0)}",
            f"- 说明：{t_trade.get('note', '')}（T+1限制）",
        ])

    lines.extend(["", _SectionDivider(), "", _NextSection(sec, "技术面信号")])
    lines.extend(_BuildSignalListSection(
        analysis.get("tech_signals", []),
        "暂无技术面信号",
    ))

    lines.extend(["", _NextSection(sec, "资金面信号")])
    lines.extend(_BuildSignalListSection(
        analysis.get("fund_signals", []),
        "暂无资金面信号",
    ))

    lines.extend(["", _NextSection(sec, "市场热点"), ""])
    concept = data.get("concept")
    if concept is not None and not concept.empty:
        sector_signals = analysis.get("sector_signals", [])
        for sig in sector_signals:
            if "概念板块" in sig:
                lines.append(f"**{sig}**")
                break

        lines.append("")
        lines.append("**概念板块 TOP5**")
        for i in range(min(5, len(concept))):
            row = concept.iloc[i]
            name = str(row.get("板块名称", ""))
            pct = float(row.get("涨跌幅", 0))
            if pct >= 0:
                pct_text = f'<font color="#E53935">+{pct:.2f}%</font>'
            else:
                pct_text = f'<font color="#43A047">{pct:.2f}%</font>'
            lines.append(f"- **{i + 1}.** {name}  {pct_text}")
    else:
        lines.append("> 板块数据暂不可用")

    lhb = data.get("lhb")
    if lhb is not None and not lhb.empty:
        lines.extend(["", "## ⚡ 龙虎榜信号"])
        for _, row in lhb.iterrows():
            date_str = str(row.get("上榜日", row.get("日期", "")))
            reason = str(row.get("上榜原因", ""))
            lines.append(f"- {date_str} {reason}")

    lines.extend([
        "",
        _SectionDivider(),
        "> ⚠️ 以上分析由程序自动生成，仅供参考，不构成投资建议。股市有风险，投资需谨慎。",
    ])

    return "\n".join(lines)


def BuildReportMarkdown(
    data: dict[str, Any],
    analysis: dict[str, Any],
    advice: dict[str, Any],
    session_label: str = "盘中",
    report_mode: str = "daily",
    full: bool = False,
    detail_url: str | None = None,
) -> str:
    """构建分析报告 Markdown；full=True 时含网页详细章节（调试用）。"""
    if full:
        dingtalk_part = BuildDingTalkReportMarkdown(
            data, analysis, advice, session_label, report_mode, detail_url=None,
        )
        detail_part = BuildDetailSectionsMarkdown(
            data, analysis, advice, session_label, report_mode,
        )
        return f"{dingtalk_part}\n\n---\n\n{detail_part}"

    return BuildDingTalkReportMarkdown(
        data, analysis, advice, session_label, report_mode, detail_url=detail_url,
    )


def BuildPortfolioReportMarkdown(portfolio_advice: dict[str, Any]) -> str:
    """构建持仓专报 Markdown。"""
    ai_tag = "（AI）" if portfolio_advice.get("llm_used") else ""
    current_price = portfolio_advice.get("current_price", 0)
    lines: list[str] = [
        f"# 💼 {config.STOCK_NAME}({config.STOCK_CODE}) 持仓建议{ai_tag}",
        f"> {NowStr()} ｜ 现价 **{FormatPrice(current_price)}**",
        "",
        _SectionDivider(),
        "",
        "## 一、整体概况",
        f"> 加权回本价 **{FormatPrice(portfolio_advice.get('overall_breakeven', 0))}**",
    ]
    recovery = portfolio_advice.get("recovery_summary", "")
    if recovery:
        lines.append(f"> {recovery}")

    accounts = portfolio_advice.get("accounts", [])
    for i, acct in enumerate(accounts, 1):
        cn = ["", "一", "二", "三", "四", "五"][i + 1] if i + 1 < 6 else str(i + 1)
        lines.extend([
            "",
            _SectionDivider(),
            "",
            f"## {cn}、{acct.get('account', f'账户{i}')}",
            f"> 成本 **{acct.get('cost', 0):.3f}** × **{acct.get('shares', 0)} 股**",
            f"> 浮盈亏 **{acct.get('pnl', 0):+.2f} 元**（{acct.get('pnl_pct', 0):+.2f}%）",
            f"> 距回本需涨 **{acct.get('distance_to_breakeven_pct', 0):+.1f}%**",
        ])
        if acct.get("strategy"):
            lines.append(f"> 策略{ai_tag}：{acct['strategy']}")
        if acct.get("action"):
            lines.append(f"> 建议操作：**{acct['action']}**")
        if acct.get("target_price"):
            lines.append(f"> 目标价 {FormatPrice(acct['target_price'])}")
        if acct.get("note"):
            lines.append(f"> {acct['note']}")
        t_trade = acct.get("t_trade") or {}
        if t_trade.get("enabled"):
            lines.append(
                f"> 做T：低吸 {FormatPrice(t_trade.get('buy_price', 0))} "
                f"/ 高抛 {FormatPrice(t_trade.get('sell_price', 0))} "
                f"/ {t_trade.get('shares', 0)}股"
            )
            if t_trade.get("note"):
                lines.append(f"> {t_trade['note']}（T+1限制）")

    lines.extend([
        "",
        _SectionDivider(),
        "> ⚠️ 以上持仓建议仅供参考，不构成投资建议。",
    ])
    return "\n".join(lines)


def SendMarkdown(title: str, text: str) -> bool:
    """发送 Markdown 消息到钉钉。"""
    if not config.DINGTALK_WEBHOOK:
        logger.warning("DINGTALK_WEBHOOK 未配置，报告输出到日志：\n%s", text)
        return False

    url = BuildSignedUrl(config.DINGTALK_WEBHOOK, config.DINGTALK_SECRET)
    payload = {
        "msgtype": "markdown",
        "markdown": {"title": title, "text": text},
    }

    try:
        resp = requests.post(url, json=payload, timeout=10)
        result = resp.json()
        if result.get("errcode") == 0:
            logger.info("钉钉推送成功")
            return True
        logger.error("钉钉推送失败: %s", result.get("errmsg", resp.text))
        return False
    except requests.RequestException as exc:
        logger.error("钉钉推送网络异常: %s", exc)
        return False


def PushReport(
    data: dict[str, Any],
    analysis: dict[str, Any],
    advice: dict[str, Any],
    session_label: str = "盘中",
    report_mode: str = "daily",
    push_time: str | None = None,
) -> bool:
    """生成并推送分析报告（钉钉精简 + 网页详细）。"""
    title = f"{config.STOCK_NAME}({config.STOCK_CODE}) {session_label}分析报告"
    rt = data.get("realtime") or {}
    change_pct = rt.get("change_pct", 0)

    detail_url: str | None = None
    detail_local_hint = ""
    if config.REPORT_WEB_ENABLED:
        detail_md = BuildDetailSectionsMarkdown(
            data, analysis, advice, session_label, report_mode,
        )
        detail_title = f"{config.STOCK_NAME}({config.STOCK_CODE}) 深度报告"
        detail_url = PublishDetailReport(
            title=detail_title,
            markdown_body=detail_md,
            meta={
                "stock_name": config.STOCK_NAME,
                "stock_code": config.STOCK_CODE,
                "session_label": session_label,
                "change_pct": change_pct,
                "generated_at": NowStr(),
            },
            session_label=session_label,
            push_time=push_time,
        )
        if detail_url:
            logger.info("详细报告链接: %s", detail_url)
            local_path = config.ResolveReportWebOutputDir() / "latest.html"
            logger.info(
                "详细报告链接已生成；jsDelivr CDN 推送后约 1~5 分钟生效，"
                "GitHub Pages 需开启 main/docs 且 docs/reports 已 push"
            )
        elif config.REPORT_WEB_LOCAL_HINT:
            local_path = config.ResolveReportWebOutputDir() / "latest.html"
            if local_path.exists():
                detail_local_hint = (
                    f"（网页已生成: `{local_path}`，请在 .env 配置 "
                    f"REPORT_WEB_BASE_URL=https://cdn.jsdelivr.net/gh/susu108/AI_Stock_Analysis@main/docs/reports）"
                )
                logger.warning(
                    "REPORT_WEB_BASE_URL 未配置或无效，钉钉不附公网链接。"
                    "推荐: https://cdn.jsdelivr.net/gh/susu108/AI_Stock_Analysis@main/docs/reports"
                )

    text = BuildDingTalkReportMarkdown(
        data, analysis, advice, session_label, report_mode,
        detail_url=detail_url,
        detail_local_hint=detail_local_hint,
    )
    return SendMarkdown(title, text)


def PushPortfolioReport(portfolio_advice: dict[str, Any]) -> bool:
    """推送持仓专报。"""
    title = f"{config.STOCK_NAME}({config.STOCK_CODE}) 分账户持仓建议"
    text = BuildPortfolioReportMarkdown(portfolio_advice)
    return SendMarkdown(title, text)


def PushErrorNotice(message: str = "核心数据获取失败，请检查网络连接") -> bool:
    """推送数据异常通知。"""
    title = f"{config.STOCK_NAME} 数据异常通知"
    text = (
        f"# ⚠️ {config.STOCK_NAME}({config.STOCK_CODE}) 数据异常\n\n"
        f"> {NowStr()}\n\n"
        f"{message}\n\n"
        f"---\n"
        f"> 请检查网络连接或稍后重试。"
    )
    return SendMarkdown(title, text)
