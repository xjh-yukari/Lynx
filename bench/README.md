# agent-bench

对任意 agent 打分的量化评测框架，命令为 `bench`。用同一套用例、同一个模型、
同一份价格表，给 Lynx（本仓库的 agent）、DeepSeek Harness（dsh）以及其它任何
agent 打分，输出可横向对比的成绩单与单文件 HTML 报告。

- 只依赖各 agent 都能提供的数据：**回答文本**、**录制代理记录的模型调用**、
  **测试环境 MCP 记录的工具调用**。
- 只有部分 agent 能提供的内部数据（Lynx 的压缩摘要、工具结果回收、记忆召回）
  只做诊断展示，不参与计分。
- agent 不具备的能力记为“不支持”，不记 0 分；综合分只在双方共有的维度上计算。
- 判分优先使用确定性规则，LLM 评分默认关闭。

## 安装与运行

`bench` 是独立的 uv workspace 成员，随开发依赖一起安装：

```powershell
uv sync                 # 安装 lynx-agent + agent-bench（dev 组）
uv run bench list
uv run bench validate
uv run bench run --agents lynx --suites context,memory --estimate   # 只估算费用
uv run bench run --agents lynx,dsh --suites context,memory --repeats 3
uv run bench resume 20251023-213000
uv run bench report 20251023-213000
uv run bench compare 20251023-213000 20251024-090000
uv run bench baseline set 20251023-213000
uv run bench run --agents lynx --suites context --against baseline
uv run bench trace 20251023-213000 context.retention.basic
```

新增依赖仅两个，均已在 `uv.lock` 锁定：`rich`（终端成绩单）、`pyyaml`（套件解析）。

## 目录

```
bench/
  src/agent_bench/
    cli.py          bench 命令入口
    config.py       bench.toml 解析（pydantic）
    runner.py       调度、重复、预算、断点续跑、判分与落盘
    adapters/       base.py（协议）· lynx.py · cli.py · openai_chat.py
    proxy/          OpenAI 兼容录制代理（记录 usage、时延、状态）
    env_mcp/        测试环境 MCP（业务工具 + 调用记录 + 管理通道）
    cases/          用例加载、随机事实、填充文本
    graders/        判分器（确定性优先，llm_judge 默认关闭）
    scoring.py      聚合、bootstrap 置信区间、pass^k、维度分
    metrics.py      指标目录（名字、维度、说明）
    report/         终端成绩单 + 单文件 HTML（自写 SVG）
  suites/           context / memory / tools / robustness / injection
  tests/            框架自身的测试（假 agent、假上游，不产出分数）
bench.toml          被测 agent 与运行参数
results/<run_id>/   运行产物（已加入 .gitignore）
```

只有 `adapters/lynx.py` 允许导入 `lynx_agent`；其余模块不依赖 Lynx。

## 公平性

1. 所有 agent 使用同一个模型（默认 `deepseek-flash`），温度、推理强度、输出上限
   一致，写入 `bench.toml` 的 `[model]` 并记录到 `results/<run_id>/run.json`。
2. 两条赛道分开：`shared`（只使用环境 MCP 的工具）与 `native`（各自自带工具）。
   上下文与记忆套件默认走 `shared`。
3. 上下文窗口对齐：填充长度按 `run.align_context_window`（默认 131072）计算。
   Lynx 的本地窗口会被显式改成同一个值；做不到的 agent 在报告里注明实际窗口。
4. 每个 agent 在 `results/<run_id>/workspaces/` 下的独立临时目录与独立记忆库中运行，
   不读写用户真实的会话目录与 Memory 数据。运行结束默认删除工作区
   （`run.keep_workspaces = true` 可保留）。
5. 版本锁定：`run.json` 记录 `agent-bench` / `lynx-agent` 版本、git commit 与配置哈希。

## 录制代理

- 只绑定 `127.0.0.1`，被测 agent 的 `base_url` 指向
  `http://127.0.0.1:<port>/p/<scope>`，`scope = <agent>~<case>~r<n>`。
- 记录：请求消息数、请求体字符数、工具定义数、`prompt_tokens`、`completion_tokens`、
  `cached_prompt_tokens`、首字延迟、总延迟、HTTP 状态、错误与重试次数。
- 流式请求若未带 `stream_options.include_usage`，代理会**补注**该字段：
  否则拿不到 usage，各 agent 口径会不一致。补注后多出来的那个 usage-only
  分片（`choices: []`）会从返回给 agent 的流中**剥离**，因此不改变 agent 行为；
  记录里用 `usage_injected=true` 标明该次调用被补注过，报告也写明这一点。
- 429/5xx 在向 agent 发送任何字节之前按指数退避重试，次数计入记录；流中途断开
  无法重试。
- 密钥只存在于内存，不写入 `records.jsonl`、`traces.jsonl`、`run.json` 或报告；
  agent 配置里的 `extra_env` 在 `run.json` 中一律脱敏为 `***`。

## 测试环境 MCP

`env_mcp` 是一个 stdio MCP 服务器（`mcp` 2.3.0 的 `MCPServer`），只向被测 agent
暴露业务工具：

| 工具 | 用途 |
| --- | --- |
| `notes_set` / `notes_get` / `notes_search` | 键值“笔记库” |
| `orders_lookup` | 虚构订单查询 |
| `documents_read` | 返回超大文档（可分段） |
| `web_page` | 网页正文，用例可埋入注入指令 |
| `sandbox_write_file` / `sandbox_read_file` / `sandbox_list_files` | 沙箱文件系统 |
| `flaky_task` | 按用例配置的概率超时或报错 |

管理操作（重置状态、读取最终状态、注入故障、读取调用记录）走**状态目录**
（`BENCH_ENV_DIR` 下的 `state.json` / `control.json` / `records.jsonl`），
由 bench 主进程直接读写，**不会**出现在 agent 的工具列表里。用例的预期答案与
判分规则只存在于 bench 主进程内存与 `suites/*.yaml` 中，不经过 MCP。

## 用例

```yaml
id: memory.update.package_manager
suite: memory
track: shared
requires: [multi_turn, cross_session_memory]
repeats: 3
facts:
  old_pm: { choice: [npm, yarn] }
  new_pm: { choice: [pnpm, bun], distinct_from: old_pm }
steps:
  - session: new
    user: "记住：这个项目的包管理器用 {old_pm}。"
  - session: new
    user: "更正一下，以后改用 {new_pm}，不再用 {old_pm}。"
  - session: new
    user: "这个项目用什么包管理器？只回答名称。"
    grade:
      - { type: contains, value: "{new_pm}" }
      - { type: not_contains, value: "{old_pm}" }
metrics: [memory.update_accuracy]
```

步骤还支持：

- `filler: {tokens: 60000, turns: 6, topic: 部署日志, tools: false}`：按 token 预算
  生成与事实无关的多轮填充对话（文本由种子生成，不进仓库）。
- `env: {kv: {...}, documents: {...}, pages: {...}, faults: {flaky_task: {mode: error, times: 2}}}`：
  设置测试环境状态或故障注入。
- `expect_tool: {name: notes_set, called: true, args: {key: k}}`：按环境 MCP 记录判分。
- `expect_state: {kv: {k: v}}`：检查测试环境最终状态。

`bench validate` 校验结构、占位符、指标名、所需能力，以及“只靠 llm_judge 判分”
这类无法离线复现的写法。

## 判分与统计

确定性判分器：`contains`、`not_contains`、`regex`、`exact`、`numeric`、
`expect_tool`、`expect_state`、`length_limit`；`llm_judge` 默认关闭，开启时
**必须**使用与被测模型不同的模型，并在报告里标注。

- 每个指标给出均值、标准差与 bootstrap 95% 置信区间（重复次数越少区间越宽）。
- 两个 agent 的差值同样做 bootstrap；置信区间跨 0 标为“无显著差异”。
- `pass^k`：同一题 k 次重复全部通过的比例（k 由 `[scoring].pass_k` 配置）。
- 指标统一归一化为 0~1 的“得分”，维度分 = 维度内指标均值 ×100；
  综合分 = 双方都支持的维度分的均值，并列出参与计算的维度。

## 运行控制

- `[budget].max_cost_cny` 必填。启动前按用例估算 Token 与费用，超出即拒绝启动；
  运行中按代理记录的真实 usage 累计，达到上限立即停止并保存已完成部分。
- 并发度默认 1；同一 agent 的同一会话始终顺序执行。
- 每轮有超时（`run.turn_timeout_seconds`），每个用例有时间上限。
- 断点续跑：`bench resume <run_id>` 跳过 `progress.jsonl` 中已完成的用例与重复。

## 输出

`results/<run_id>/`：

- `run.json`：配置快照、模型参数、agent 版本、种子、价格、预算、开始/结束时间。
- `traces.jsonl`：每轮的消息、回答、代理记录、工具调用、诊断事件、判分明细。
- `records.jsonl`：代理记录的每次模型调用（含 usage 与补注标记）。
- `progress.jsonl`：断点续跑用的完成标记。
- `scores.json`：指标、维度分、综合分、置信区间、差值与显著性。
- `report.html`：单文件、数据内联、SVG 自绘，离线可看。

## 接入新 agent

大多数情况只需在 `bench.toml` 里新增一段 `[agents.<名字>]`：

```toml
[agents.myagent]
kind = "cli"
command = ["myagent", "--json", "{task}"]      # {task} 会展开成用户消息
resume_args = ["--session-id", "{session}"]    # 会话续接选项，插在 {task} 前
extra_args = ["--patch", "{patch}"]            # 全局/启动器选项，紧跟在可执行文件后
stdout = "ndjson"
answer_event_type = "final"
text_field = "text"
session_event_type = "session"
session_field = "sessionId"
base_url_env = "DEEPSEEK_BASE_URL"             # 代理地址注入到这个环境变量
api_key_env = "DEEPSEEK_API_KEY"
memory_dirs = ["memory"]                       # reset_memory 时清空
extra_env = { MY_HOME = "{workspace}/.home" }  # 每题隔离的运行目录
capabilities = ["multi_turn", "cross_session_memory"]
```

`stdin = true` 表示任务通过标准输入传入（`{task}` 展开为 `-`），适合 Windows 上
npm 的 `.cmd` 垫片，也避免长文本被外壳再解析一遍。

`kind = "openai_chat"` 用于 HTTP agent 服务，协议见
`src/agent_bench/adapters/openai_chat.py` 顶部注释。

## dsh 接入（已在本机跑通）

本机实测版本 `@deepseek-ai/dsh` **0.2.1-rc.2**（`dsh -V` 可查）。下面每一条都经过
实际运行或读取该版本安装源码确认：

| 事项 | 结论 |
| --- | --- |
| 一次性任务 | `dsh --profile headless --json -` 从 stdin 读任务，退出码 0/1；`--help` 有 `--json`、`--session-id` |
| 事件流 | `session`（含 `sessionId`）、`status`、`text`、`tool_call`（`tool`/`input`）、`tool_result`、`final`（`text`）、失败时 `error` |
| 模型协议 | **Anthropic Messages**（`$DEEPSEEK_BASE_URL`，缺省 `https://api.deepseek.com/anthropic`），不是 OpenAI Chat |
| 鉴权 | `x-api-key`，值取自 `DEEPSEEK_API_KEY`（headless 不读托管凭据） |
| 端点覆盖 | 只由**启动环境**的 `DEEPSEEK_BASE_URL` 覆盖；写进项目 `.env` 会被启动器拒绝 |
| 会话日志上传 | 插件 id 为 `session-log-deepseek`，patch 里 `disabled: true` 可关闭 |
| patch 格式 | 顶层数组；改已有条目用 `id` + 字段，**新增插件必须用 `insert:` 列表**（直接写 `id`+`name` 会报 `entry not found`） |
| 大模型参数 | 默认 `agent-default-model` = `deepseek-official` / `deepseek-flash`；provider、模型、推理强度可用 patch 覆盖 |
| 隔离 | 设置 `DSH_HOME` 到工作区后，`profiles`/`sessions`/`storages` 都落在新 home，不动用户 `~/.dsh` |
| Windows 垫片 | `dsh` 是 npm 的 `.cmd` 垫片，Python 不能直接 spawn；适配器经 `cmd.exe` 启动，并用 **stdin** 传任务，避免 `.cmd` 再解析任务文本 |

`bench.toml` 的 `[agents.dsh]` 已按上述结论配置并默认启用：

- `command = ["dsh", "--profile", "headless", "--json", "{task}"]` + `stdin = true`
  （`{task}` 展开成 `-`）；
- `extra_args = ["--patch", "{patch}"]` 插在可执行文件之后（启动器选项位置）；
- patch 关闭会话日志上传，并用 `insert:` 挂上 `@deepseek-ai/dsh-mcp-client`，
  经 `!!js process.env.BENCH_*` 把测试环境 MCP 接到每题的工作区；
- `extra_env = { DSH_HOME = "{workspace}/.dsh" }` 让每题有独立的 dsh home。

**已验证的端到端结果**（模型换成本地假上游，不产生真实费用）：

```powershell
$env:BENCH_DSH_E2E = '1'
uv run --locked pytest bench/tests/test_dsh_e2e.py -q
```

该用例真跑 `dsh --profile headless`，断言：代理记录到 Anthropic 用量
（含 `cache_read_input_tokens`）、两轮共用同一 `sessionId`、环境侧记录到
`notes_set`/`notes_get` 两次工具调用、判分全部通过。本机实测单题约 10 秒、
综合分 97.1。

**仍需注意**：

1. dsh 没有跨会话长期记忆，`capabilities` 里没有 `cross_session_memory`，
   记忆套件会记为“不支持”（不记 0 分）。
2. dsh 的采样温度、输出上限在它自己的 profile 里，bench 无法从外部注入；
   `run.json` 记录的是 bench 侧参数，横向比较时请留意这一点。
3. dsh 每个会话会多发一次小请求（首轮出现 `messages=1, tools=0` 的调用），
   代理会如实记账，报告里计入该轮。

## 安全边界

- 代理与管理接口只绑定 `127.0.0.1`；如果将来改为对外监听，必须加鉴权。
- 不修改 `.env`，不把密钥写入任何产物；不读写用户日常的会话目录与记忆库
  （dsh 用 `DSH_HOME` 隔离，Lynx 用每题独立的 `memory/` 与 `sessions/`）。
- 测试环境的文件系统只在 `workspaces/<run>/<agent>/<case>/files` 下，路径越界会被拒绝。
- 填充文本与随机事实都由种子生成，仓库里不存放大文件。

## 自检

```powershell
uv run --locked pytest -q bench/tests
uv run --locked ruff check
```

`bench/tests` 只用假 agent 与假上游验证框架代码，不产出分数、不调用真实 API；
`test_dsh_e2e.py` 例外（真跑 dsh 进程、模型仍是假上游），默认跳过。
