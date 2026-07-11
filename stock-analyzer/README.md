# 股票分析钉钉定时推送工具

每天自动获取目标股票实时行情、K线、板块热点、资金流向、最新资讯，进行 AI 综合分析，生成买卖建议，通过钉钉机器人推送到群。

## 快速开始

```bash
cd stock-analyzer
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env 填入钉钉 Webhook、Secret、DeepSeek Key、持仓信息

# 模拟数据测试（无需网络）
python mock_test.py

# 日常分析（无持仓内容）
python main.py --now

# 分账户持仓建议
python main.py --portfolio
```

## 配置说明

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `DINGTALK_WEBHOOK` | 钉钉机器人 Webhook URL | 必填 |
| `DINGTALK_SECRET` | 加签密钥（SEC开头） | 可选 |
| `STOCK_CODE` | 股票代码 | 301075 |
| `STOCK_NAME` | 股票名称 | 多瑞生物 |
| `PUSH_TIMES` | 多次推送，逗号分隔 | 09:00,11:30,14:30 |
| `POSITIONS` | 分账户持仓 JSON | 见 .env.example |
| `MANUAL_RUN` | true=立即日常分析 | true |
| `LLM_ENABLED` | 是否启用大模型 | true |
| `WEB_SEARCH_ENABLED` | 是否启用联网搜索（涨跌归因） | false |
| `WEB_SEARCH_PROVIDER` | 搜索 Provider：`tavily` / `serper` | tavily |
| `WEB_SEARCH_API_KEY` | Tavily 或 Serper API Key | 可选 |
| `STOCK_THEMES` | 个股题材词（逗号分隔，用于搜索） | 空 |
| `PRICE_MOVE_MIN_PCT` | 涨跌幅低于该值输出简版归因 | 2.0 |
| `REPORT_WEB_ENABLED` | 是否生成网页详细报告并在钉钉附链接 | true |
| `REPORT_WEB_BASE_URL` | GitHub Pages 报告根 URL | 空 |
| `REPORT_WEB_OUTPUT_DIR` | HTML 输出目录（相对仓库根） | docs/reports |
| `REPORT_WEB_RETENTION` | 保留历史报告份数 | 30 |

## 涨跌归因拆解

**涨跌归因拆解、详细依据、资讯解读** 发布在网页详细报告中；钉钉推送精简版，并附链接：

```markdown
- **完整深度报告** [涨跌归因 · 详细依据 · 资讯解读](https://你的用户名.github.io/AI_Stock_Analysis/reports/latest.html)
```

网页报告五维涨跌归因（对标豆包粒度）：

1. **核心直接催化** — 政策/资讯 + 题材匹配
2. **板块资金共振** — 板块领涨、高低切换
3. **技术面超跌修复** — 箱体破位、压力转换
4. **市场情绪与筹码** — 量能、小盘优势、融资盘
5. **关键利空/隐患** — 利好兑现、套牢区、杠杆抛压

启用联网搜索可补全官方政策细节（如基药目录、具体药品）：

```env
WEB_SEARCH_ENABLED=true
WEB_SEARCH_PROVIDER=tavily
WEB_SEARCH_API_KEY=tvly-xxx
STOCK_THEMES=GLP-1,司美格鲁肽,多肽原料药
```

[Tavily](https://tavily.com/) 或 [Serper](https://serper.dev/) 注册获取 API Key。

## 网页详细报告（GitHub Pages）

1. 仓库 **Settings → Pages → Build and deployment → Branch: main → Folder: /docs**
2. Actions Variables 设置：
   - `REPORT_WEB_ENABLED=true`
   - `REPORT_WEB_BASE_URL=https://<用户名>.github.io/<仓库名>/reports`
3. 每次推送后 Actions 自动将 HTML 写入 `docs/reports/` 并 commit

本地调试时 `REPORT_WEB_BASE_URL` 留空仍会生成 `docs/reports/latest.html`，但不会出现在钉钉链接中。

### 日常分析（默认）

```bash
python main.py --now
```

- 实时行情 + 三维度评分 + 买卖建议（精简版）
- 钉钉附 **网页深度报告** 链接（涨跌归因 / 详细依据 / 资讯解读）
- 资讯 AI 过滤，突出涨/跌影响
- **不含**持仓、回本、做T建议
- 报告末尾提示如何获取持仓建议

### 分账户持仓建议

```bash
python main.py --portfolio
```

- 按 `POSITIONS` 中每个账户独立分析
- 含回本路径、减亏/做T策略
- 仅推送持仓专报，不含日常分析

## 定时推送

```env
MANUAL_RUN=false
```

```bash
nohup python3 main.py > stock_analyzer.log 2>&1 &
```

定时任务仅推送**日常分析**（不含持仓）。持仓建议需手动执行 `--portfolio`。

## GitHub Actions 定时推送

本地 `nohup` 需电脑常开；可将代码推到 GitHub，由 Actions 在云端按北京时间定时执行。

### 1. 创建 GitHub 仓库并推送

```bash
cd /path/to/AI_Stock_Analysis
git init
git add .
git commit -m "feat: stock analyzer with GitHub Actions scheduled push"
git branch -M main
git remote add origin https://github.com/你的用户名/你的仓库名.git
git push -u origin main
```

### 2. 配置 Secrets / Variables

仓库 **Settings → Secrets and variables → Actions**：

| 类型 | 名称 | 说明 |
|------|------|------|
| Secret | `DINGTALK_WEBHOOK` | 钉钉 Webhook（必填） |
| Secret | `DINGTALK_SECRET` | 加签密钥（可选） |
| Secret | `DEEPSEEK_API_KEY` | DeepSeek API Key |
| Secret | `WEB_SEARCH_API_KEY` | Tavily/Serper Key（涨跌归因联网搜索，可选） |
| Secret | `POSITIONS` | 持仓 JSON（仅 `--portfolio` 需要，日常推送可留空 `[]`） |
| Variable | `STOCK_CODE` | 可选，默认 301075 |
| Variable | `STOCK_NAME` | 可选，默认 多瑞生物 |
| Variable | `STOCK_THEMES` | 可选，如 `GLP-1,司美格鲁肽,多肽原料药` |
| Variable | `WEB_SEARCH_ENABLED` | 可选，`true` 启用联网搜索 |
| Variable | `REPORT_WEB_ENABLED` | 可选，`true` 生成网页报告 |
| Variable | `REPORT_WEB_BASE_URL` | 可选，GitHub Pages 报告 URL 前缀 |
| Variable | `LLM_ENABLED` | 可选，`true` / `false` |

### 3. 调度时间

工作流文件：`.github/workflows/stock-analyzer-push.yml`

| 北京时间 | 时段标签 |
|----------|----------|
| 09:00 | 开盘前 |
| 11:30 | 午盘 |
| 14:30 | 尾盘 |

每日三个时间点触发（含周末与法定节假日）。程序内部会判断是否为交易日，非交易日自动跳过。

### 4. cron-job.org 定时触发（推荐，比 GitHub 内置 schedule 更稳定）

GitHub Actions 内置 `schedule` 在免费仓库可能延迟数分钟甚至漏跑。改用 [cron-job.org](https://cron-job.org) 通过 API 触发 workflow，稳定性更好。

#### 4.1 创建 GitHub Personal Access Token

1. 打开 [GitHub → Settings → Developer settings → Personal access tokens](https://github.com/settings/tokens)
2. 新建 **Fine-grained token**（或 Classic token）：
   - 仓库权限：`susu108/AI_Stock_Analysis`（或你的仓库）
   - 勾选 **Actions: Read and write**
   - 勾选 **Contents: Read and write**（用于推送报告到 GitHub Pages）
3. 复制生成的 token（形如 `github_pat_xxx` 或 `ghp_xxx`），妥善保存

#### 4.2 在 cron-job.org 创建 3 个 Cron Job

注册并登录 [cron-job.org](https://console.cron-job.org/)，分别创建以下任务（**时区选 `Asia/Shanghai`**）：

| 任务名 | Cron 表达式 | push_time |
|--------|-------------|-----------|
| 股票分析-开盘前 | `0 9 * * *` | `09:00` |
| 股票分析-午盘 | `30 11 * * *` | `11:30` |
| 股票分析-尾盘 | `30 14 * * *` | `14:30` |

每个任务的 HTTP 请求配置如下：

| 字段 | 值 |
|------|-----|
| URL | `https://api.github.com/repos/susu108/AI_Stock_Analysis/dispatches` |
| Method | `POST` |
| Request body | 见下方 JSON（按任务替换 `push_time`） |
| Request timeout | `30` 秒 |

**Request body（JSON）：**

```json
{
  "event_type": "stock-push",
  "client_payload": {
    "push_time": "09:00"
  }
}
```

**Request headers（添加 3 个）：**

| Header | Value |
|--------|-------|
| `Accept` | `application/vnd.github+json` |
| `Authorization` | `Bearer 你的GitHub_TOKEN` |
| `X-GitHub-Api-Version` | `2022-11-28` |
| `Content-Type` | `application/json` |

> 若仓库名不是 `susu108/AI_Stock_Analysis`，请替换 URL 中的 owner/repo。

#### 4.3 验证

1. 在 cron-job.org 点击 **Run now** 手动触发一次
2. 打开 GitHub 仓库 **Actions** 页，应看到 `Stock Analyzer Push` 正在运行
3. 确认钉钉收到推送

#### 4.4 注意事项

- 工作流已默认**注释掉** GitHub 内置 `schedule`，避免与 cron-job.org 重复触发
- cron-job.org 免费版足够使用；建议开启失败通知邮件
- Token 仅保存在 cron-job.org 的 Header 中，不要提交到代码仓库

### 5. 手动测试

GitHub 仓库 **Actions → Stock Analyzer Push → Run workflow**，可选推送时刻做测试。

本地等价命令：

```bash
python main.py --scheduled --push-time 09:00
```

## 免责声明

以上分析由程序自动生成，仅供参考，不构成投资建议。股市有风险，投资需谨慎。
