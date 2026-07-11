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

## 两种推送模式

### 日常分析（默认）

```bash
python main.py --now
```

- 实时行情 + 三维度评分
- 资讯 AI 过滤，突出涨/跌影响
- 政策/监管资讯完整展示
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

## GitHub Actions 定时推送（推荐）

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
| Secret | `POSITIONS` | 持仓 JSON（仅 `--portfolio` 需要，日常推送可留空 `[]`） |
| Variable | `STOCK_CODE` | 可选，默认 301075 |
| Variable | `STOCK_NAME` | 可选，默认 多瑞生物 |
| Variable | `LLM_ENABLED` | 可选，`true` / `false` |

### 3. 调度时间

工作流文件：`.github/workflows/stock-analyzer-push.yml`

| 北京时间 | 时段标签 |
|----------|----------|
| 09:00 | 开盘前 |
| 11:30 | 午盘 |
| 14:30 | 尾盘 |

仅周一到周五触发；法定节假日会在程序内 `IsTradingDay` 自动跳过。

### 4. 手动测试

GitHub 仓库 **Actions → Stock Analyzer Push → Run workflow**，可选推送时刻或勾选「忽略交易日」做测试。

本地等价命令：

```bash
python main.py --scheduled --push-time 09:00
```

## 免责声明

以上分析由程序自动生成，仅供参考，不构成投资建议。股市有风险，投资需谨慎。
