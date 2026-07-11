"""网页详细报告 — HTML 渲染与 GitHub Pages 发布。"""

from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import markdown

import config
from utils import NowStr, SetupLogger

logger = SetupLogger(config.LOG_LEVEL)

_DISCLAIMER = (
    "以上分析由程序自动生成，仅供参考，不构成投资建议。股市有风险，投资需谨慎。"
)


def PrepareWebMarkdown(markdown_body: str) -> str:
    """将钉钉风格 Markdown 预处理为更适合网页渲染的格式。"""
    lines = markdown_body.split("\n")
    result: list[str] = []
    skip_header = True

    for line in lines:
        stripped = line.strip()

        if skip_header:
            if not stripped:
                continue
            if stripped.startswith("# ") or (
                stripped.startswith("> ") and "｜" in stripped
            ):
                continue
            skip_header = False

        prev = result[-1].strip() if result else ""
        if (
            stripped.startswith("- ")
            and prev
            and not prev.startswith("- ")
            and not prev.startswith("*")
            and not prev.startswith("<h")
            and prev != "---"
            and result[-1] != ""
        ):
            result.append("")

        result.append(line)

    return "\n".join(result)


def BuildReportFileName(
    stock_code: str,
    session_label: str,
    push_time: str | None = None,
) -> str:
    """生成报告 HTML 文件名，如 301075_20260711_0930_开盘前.html。"""
    now = datetime.now()
    time_part = (push_time or now.strftime("%H:%M")).replace(":", "")
    date_part = now.strftime("%Y%m%d")
    safe_label = re.sub(r"[^\w\u4e00-\u9fff-]", "", session_label) or "report"
    return f"{stock_code}_{date_part}_{time_part}_{safe_label}.html"


def RenderDetailReportHtml(
    title: str,
    markdown_body: str,
    meta: dict[str, Any] | None = None,
) -> str:
    """将 Markdown 正文渲染为完整 HTML 页面。"""
    meta = meta or {}
    stock_name = str(meta.get("stock_name", config.STOCK_NAME))
    stock_code = str(meta.get("stock_code", config.STOCK_CODE))
    session_label = str(meta.get("session_label", ""))
    generated_at = str(meta.get("generated_at", NowStr()))
    change_pct = meta.get("change_pct")

    nav_items = [
        ("section-catalyst", "涨跌归因拆解"),
        ("section-reasons", "详细依据"),
        ("section-news", "资讯解读"),
    ]
    nav_html = "".join(
        f'<a href="#{anchor}">{label}</a>' for anchor, label in nav_items
    )

    pct_badge = ""
    if change_pct is not None:
        try:
            pct_val = float(change_pct)
            color = "#E53935" if pct_val >= 0 else "#43A047"
            sign = "+" if pct_val >= 0 else ""
            pct_badge = (
                f'<span class="pct-badge" style="color:{color}">'
                f"{sign}{pct_val:.2f}%</span>"
            )
        except (TypeError, ValueError):
            pct_badge = ""

    body_html = markdown.markdown(
        PrepareWebMarkdown(markdown_body),
        extensions=["extra", "nl2br", "sane_lists"],
        output_format="html5",
    )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <style>
    :root {{
      --bg: #f6f8fa;
      --card: #ffffff;
      --text: #1f2328;
      --muted: #656d76;
      --border: #d8dee4;
      --accent: #0969da;
      --up: #E53935;
      --down: #43A047;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
        "Helvetica Neue", Arial, "PingFang SC", "Microsoft YaHei", sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.65;
      font-size: 15px;
    }}
    .container {{
      max-width: 820px;
      margin: 0 auto;
      padding: 16px 16px 48px;
    }}
    header {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 20px;
      margin-bottom: 16px;
    }}
    header h1 {{
      margin: 0 0 8px;
      font-size: 1.35rem;
    }}
    .meta {{
      color: var(--muted);
      font-size: 0.9rem;
    }}
    .pct-badge {{
      font-weight: 700;
      margin-left: 8px;
    }}
    nav {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 16px;
    }}
    nav a {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 6px 14px;
      color: var(--accent);
      text-decoration: none;
      font-size: 0.88rem;
    }}
    main {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 20px;
    }}
    main h2 {{
      margin-top: 28px;
      padding-top: 8px;
      border-top: 1px solid var(--border);
      font-size: 1.15rem;
    }}
    main h2:first-child {{ border-top: none; margin-top: 0; }}
    main h2 + ul {{ margin-top: 8px; }}
    main ul, main ol {{ padding-left: 1.25rem; margin: 8px 0 16px; }}
    main li {{ margin: 6px 0; }}
    main li ul {{ margin-top: 4px; margin-bottom: 4px; }}
    main p {{ margin: 10px 0; }}
    main p > strong:first-child {{
      display: block;
      margin: 16px 0 8px;
      font-size: 1.02rem;
    }}
    main font {{ font-size: inherit; }}
    main blockquote {{
      margin: 12px 0;
      padding: 8px 12px;
      border-left: 4px solid var(--border);
      color: var(--muted);
      background: #f6f8fa;
    }}
    footer {{
      margin-top: 16px;
      color: var(--muted);
      font-size: 0.85rem;
      text-align: center;
    }}
    hr {{ border: none; border-top: 1px solid var(--border); margin: 20px 0; }}
    @media (max-width: 600px) {{
      .container {{ padding: 12px 12px 32px; }}
      header, main {{ padding: 14px; }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>{title}{pct_badge}</h1>
      <div class="meta">
        {stock_name}({stock_code}) ｜ {session_label} ｜ 生成于 {generated_at}
      </div>
    </header>
    <nav>{nav_html}</nav>
    <main>{body_html}</main>
    <footer><p>{_DISCLAIMER}</p></footer>
  </div>
</body>
</html>"""


def WriteDetailReport(
    html: str,
    output_dir: Path,
    filename: str,
) -> Path:
    """写入 HTML 文件并更新 latest.html。"""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / filename
        target.write_text(html, encoding="utf-8")
        latest = output_dir / "latest.html"
        shutil.copy2(target, latest)
        logger.info("详细报告已写入: %s", target)
        return target
    except OSError as exc:
        logger.error("写入详细报告失败: %s", exc)
        raise


def CleanupOldReports(output_dir: Path, retention: int) -> None:
    """按修改时间保留最近 N 份历史报告（不含 latest/index）。"""
    if retention <= 0 or not output_dir.exists():
        return
    try:
        candidates = [
            p for p in output_dir.glob("*.html")
            if p.name not in ("latest.html", "index.html")
        ]
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for old_file in candidates[retention:]:
            old_file.unlink(missing_ok=True)
            logger.debug("已清理旧报告: %s", old_file.name)
    except OSError as exc:
        logger.warning("清理旧报告失败: %s", exc)


def IsValidReportWebBaseUrl(base_url: str) -> bool:
    """判断是否为可用的公网报告 URL 前缀。"""
    base = base_url.strip().rstrip("/")
    if not base.startswith(("http://", "https://")):
        return False
    invalid_tokens = ("你的", "example", "placeholder", "用户名", "仓库名", "Pages地址")
    return not any(token in base for token in invalid_tokens)


def BuildDetailPublicUrl(filename: str) -> str | None:
    """根据 BASE_URL 构建公网访问链接。"""
    base = config.ResolveReportWebBaseUrl().strip().rstrip("/")
    if not IsValidReportWebBaseUrl(base):
        return None
    url = f"{base}/{filename}"
    if "cdn.jsdelivr.net" in base or "ghfast.top" in base or "ghproxy" in base or "statically.io" in base:
        url = f"{url}?v={datetime.now().strftime('%Y%m%d%H%M')}"
    return url


def EnsureGithubPagesArtifacts(output_dir: Path) -> None:
    """确保 GitHub Pages 所需文件存在（.nojekyll、根 index 跳转）。"""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / ".nojekyll").touch(exist_ok=True)
        (output_dir.parent / ".nojekyll").touch(exist_ok=True)
        index_path = output_dir.parent / "index.html"
        if not index_path.exists() or index_path.stat().st_size < 100:
            index_path.parent.mkdir(parents=True, exist_ok=True)
            index_path.write_text(
                """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="refresh" content="0; url=reports/latest.html">
  <title>AI Stock Analysis 报告</title>
</head>
<body>
  <p>正在跳转到最新分析报告… <a href="reports/latest.html">点此打开</a></p>
</body>
</html>
""",
                encoding="utf-8",
            )
            logger.info("已写入 GitHub Pages 入口: %s", index_path)
    except OSError as exc:
        logger.warning("写入 GitHub Pages 辅助文件失败: %s", exc)


def PublishDetailReport(
    title: str,
    markdown_body: str,
    meta: dict[str, Any] | None = None,
    session_label: str = "盘中",
    push_time: str | None = None,
) -> str | None:
    """发布详细报告 HTML，返回可访问 URL（无 BASE_URL 时返回 None）。"""
    if not config.REPORT_WEB_ENABLED:
        return None
    if not markdown_body.strip():
        logger.warning("详细报告正文为空，跳过发布")
        return None

    try:
        output_dir = config.ResolveReportWebOutputDir()
        EnsureGithubPagesArtifacts(output_dir)
        filename = BuildReportFileName(config.STOCK_CODE, session_label, push_time)
        html = RenderDetailReportHtml(title, markdown_body, meta)
        WriteDetailReport(html, output_dir, filename)
        CleanupOldReports(output_dir, config.REPORT_WEB_RETENTION)
        public_url = BuildDetailPublicUrl("latest.html")
        if public_url:
            logger.info("详细报告 URL: %s", public_url)
        elif config.REPORT_WEB_LOCAL_HINT:
            local_path = output_dir / "latest.html"
            logger.info("详细报告已保存本地: %s（未配置 REPORT_WEB_BASE_URL）", local_path)
        return public_url
    except Exception as exc:
        logger.error("发布详细报告异常: %s", exc)
        return None
