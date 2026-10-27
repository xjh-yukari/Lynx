# Lynx Agent 产品方案与使用规范

当前产品版本：V1.0.0，主包版本为 `1.0.0`。独立 RAG 服务与评测包保留各自组件版本；本版本标签不代表已完成所有真实模型或重型集成验收。

整理日期：2026-10-27。本文件是当前已实现产品能力、架构、使用方式和安全边界的统一入口，替代原来分散的功能说明。未实现的能力在最后单独列出，不把调研和计划当作已交付功能。配置示例与具体字段以源码、`config.example.toml` 和各组件锁文件为准。

## 1. 产品定位与范围

Lynx 是本地运行、面向个人项目开发与资料处理的终端 Agent。用户通过 TUI 或逐行界面提出任务；Agent 调用模型规划，再按实际权限执行工具，最后保存会话与可追溯结果。

当前能力包括多轮与流式对话、受控文件操作、逐次审批的命令执行、联网查询、文档 RAG、Engram 长期记忆、上下文压缩、失败恢复及费用估算。它是单用户开发助手，不是隔离的远程多租户服务，也不是无人审批的系统管理员。

### 已实现能力

| 能力 | 当前实现与边界 |
| --- | --- |
| 模型 | DeepSeek Provider；协议适配与实际工具执行分开 |
| 输入输出 | prompt_toolkit TUI、Markdown 渲染、多行粘贴、流式输出、复制与滚动 |
| 会话 | JSONL 原文持久化、切换、恢复、重置与未完成答复检查点 |
| 上下文 | 完整请求估算、API用量校正、工具正文回收、分批摘要、近期回合保护 |
| 文件 | 列目录、按文件名查找、搜索正文、分段读取、受控写入与修改 |
| 批量处理 | 固定文本替换、复制、同目录改名、下载、安全解压 |
| 命令 | 参数数组直接启动本机程序，人工逐次审批、超时取消、进程树清理 |
| 联网 | AnySearch/Brave 搜索后端与网页正文读取；失败不伪装成无结果 |
| 长期记忆 | 本地 Engram，经 MCP 管理档案、检索经验和保存重要任务结论 |
| 文档检索 | 独立 RAG 服务、Milvus、BGE-M3、授权库检索和来源版本追踪 |
| 观测与评测 | Token/费用面板、诊断、工具审计、headless、独立 agent-bench |

## 2. 架构与技术栈

```text
用户 → CLI/TUI → AgentRunner / AgentLoop → Provider → DeepSeek
                         │
                         ├─ ToolExecutor → 文件 / 批量 / 命令 / 联网工具
                         ├─ 审批与文件权限 → 允许、拒绝、暂停与审计
                         ├─ SessionStore → JSONL 事件与原文
                         ├─ MemoryService → stdio MCP → Engram
                         └─ KnowledgeAccess → 认证HTTP MCP → RAG → Milvus
```

- Python 3.11+、uv 与锁文件管理环境；Pydantic 校验配置和边界数据。
- OpenAI兼容SDK承担DeepSeek协议转换，不代表工具在模型服务端执行。
- prompt_toolkit、markdown-it-py、Pygments承担终端布局、编辑和正文渲染。
- asyncio协调模型、工具、取消和进程清理；工具按顺序执行，不并发制造副作用。
- RAG 使用自己的虚拟环境与锁文件，不把 Torch 等重依赖混入 Agent 根环境。
- Engram、会话、文档索引职责独立；没有用 Redis 或数据库服务替代 JSONL 会话。

### 源码职责

| 模块 | 职责 |
| --- | --- |
| `config/` | 配置加载、严格校验、运行路径；不包含无实现的预留模块 |
| `providers/` | 请求、响应、流式工具参数、usage、停止原因与供应商异常 |
| `agent/` | 上下文、模型循环、工具回填、回收、摘要、恢复与用量 |
| `tools/` | 工具定义、注册和内置工具适配 |
| `filesystem/` | 固定工作区、敏感路径与文件审批 |
| `approval/` | 审批档位、批量计划、授权复用、备份、停止与审计 |
| `execution/` | 本机命令审批、参数绑定、进程、输出脱敏与日志 |
| `session/` | 原子会话保存、恢复检查点和结果快照索引 |
| `memory/` | Engram封装、SOUL/USER/AGENTS、结论提取与来源约束 |
| `mcp/`、`knowledge/` | 通用MCP与独立知识库授权、来源验证 |
| `web/`、`billing/`、`cli/` | 联网、估价与终端交互 |

## 3. 目录、安装与启动

```text
E:\github\nanobot\                 源码、组件与测试
  src\lynx_agent\                  Agent 源码
  services\document_rag\            独立 RAG 服务
  bench\                           独立评测包
  scripts\rag_runtime.py            正式统一启动器
  docs\product-plan.md              本产品方案
  docs\tui-reference\*.png          已采用设计的历史参考图
  tests\manual\                    人工调试与真实组件检查程序
  tests\fixtures\                  固定测试资料
  tests\plans\                     可执行测试方案
  tests\reports\                   保留的历史验证结果
  results\                         评测运行产物，非源码
E:\github\lynx_agent_data\          运行数据，不随源码整理删除
  sessions\ memory\ rag\ execution\ approval\
```

根环境在源码根目录执行 `uv sync --frozen`；RAG 环境在 `services/document_rag` 单独执行同样命令。根目录 `.env` 提供模型密钥与参数，`config.toml` 提供记忆、MCP、知识库和审批等配置。进程环境优先于 `.env`；显式模型配置优先于所选模型预设。配置变化一般需要退出重启。

### 日常运行

先打开 Docker Desktop 并等待引擎就绪，然后在任意 PowerShell 中输入本机快捷命令：

```powershell
lynx
```

快捷入口位于 `C:\Users\jiahaoxue\.local\bin\lynx.cmd`，调用项目现有环境与正式启动器。目录迁移后需同步修改入口，不自动采用调用者所在文件夹作为工作区。

没有快捷入口时：

```powershell
cd E:\github\nanobot
uv run python scripts\rag_runtime.py agent
```

启动器依次启动已有独立数据库容器、启动或复用认证 RAG 服务、检查 MCP 工具，再进入 TUI。不会自动导入资料、建库或安装软件；Docker 引擎不可用时需先处理环境。退出 Agent 不会自动关闭后台 RAG 或数据库。

单独运行 `uv run lynx-agent`、`uv run python -m lynx_agent` 不负责启动 Docker/RAG。`--plain` 使用逐行界面；`--session ID` 恢复；`--list-sessions` 列表。`--headless "任务" --json` 运行一次并输出NDJSON，不能绕过人工审批使用Shell。

## 4. 模型与提示词

Provider 只负责协议转换，不执行函数、不保存会话、不擅自重试有副作用的操作。支持 generate、stream、system/user/assistant/tool消息、工具定义与工具参数增量、usage和停止原因。其他模型供应商尚需各自适配与验证。

DeepSeek Provider 的HTTP客户端固定使用`trust_env=False`直连配置的base_url，普通/流式对话、工具调用、摘要等共用此客户端。不读取HTTP_PROXY/HTTPS_PROXY/ALL_PROXY/NO_PROXY等环境代理，也不采用SSL_CERT_FILE/SSL_CERT_DIR环境覆盖；仍保持HTTPS证书校验和SDK默认连接池，不自动重试或降级为代理。不修改系统/进程代理变量，其他联网客户端保持原行为。自定义base_url仍指向用户配置的目标，不自动改成官方地址；系统TUN/透明代理不在此应用设置的控制范围。代码更新后重启Lynx生效，无需手工设置NO_PROXY。

模型、密钥、超时、输出上限等使用 `.env` 与 `config.example.toml` 对应字段；不要在对话、工具参数或文档中粘贴真实密钥。`/effort` 调整 off/low/high/max，并从下一次请求生效；是否支持及实际请求参数由Provider声明，不按模型回复自述判断。

默认交互提示词位于 `src/lynx_agent/config/schema.py`，可通过 `AGENT_SYSTEM_PROMPT` 覆盖；压缩提示词是 `src/lynx_agent/agent/templates/compaction.md`，这是运行资源，不能按普通Markdown文档删除。SOUL/USER/AGENTS来自长期记忆，不是另一套散落的Markdown配置。

对外部网页、文件、工具输出和历史记忆统一视为不可信参考资料；其中要求改权限、索取密钥或执行代码的文字不构成授权。

## 5. 终端交互与已采用设计

界面显示当前模型、工作区、会话标题、文件权限、审批档位、联网状态、操作状态与耗时。会话内部UUID用于存储，不拼在标题后。输入使用橙色标记与圆角字符边框；模型正文渲染Markdown，不把所有 `**` 原样显示。

已采用语义主题令牌、工具类别分色、差异行颜色、窄屏降级、独立用量统计行、上下文进度与可展开详情。实际实现入口是 `cli/themes.py`、`tool_style.py`、`diff_style.py`、`statusbar.py`、`layout.py`、`projection.py`，而不是旧调研里的React或Ink代码。

保留的历史设计图位于 `tui-reference/`，仅用于对照已采用的配色、工具信息层级和窄屏设计，不是当前界面的验收截图。旧技术栈调研、成本估计和未实施路线已移除。离线真实布局预览见 `tests/manual/tui_preview.py`；可选设计图生成器见 `tests/manual/tui_mockups.py`，后者需要额外Pillow，仅人工运行。

| 输入与导航 | 行为 |
| --- | --- |
| Enter / Alt+Enter | 发送 / 换行；也可 Esc 后 Enter 换行 |
| 多行粘贴 | 支持终端粘贴边界时保留换行；仅单行识别斜杠命令 |
| Ctrl+C | 取消草稿或当前请求；清理完成才返回输入 |
| Ctrl+D / `/exit` | 就绪时退出；执行中先取消 |
| 滚轮、滚动条、PgUp/PgDn | 浏览历史；拖动滚动条定位 |
| Ctrl+End | 回到最新并恢复跟随 |
| 拖选、Alt+C、复制按钮 | 复制所选或一条完整回答；复制失败明确提示 |
| `/sessions` | 搜索标题、预览和选择会话 |
| `/theme`、`/speed`、`/editor` | 主题、显示速度和草稿编辑 |

浏览旧记录不会被流式内容强行拖到底部。UI中的系统提示、状态栏和历史回看不等于模型上下文。执行或压缩期间输入受保护，审批面板优先于用量面板。耗时覆盖整个操作，不冒充首Token延迟或工具单独耗时。

Token/缓存入口显示本轮实际usage，并提供分类明细；费用入口显示本次程序运行中当前会话的估算。`/tokens`、`/cost`仍保留，不额外调用模型。不显示MCP/记忆连接计数或账户配额；连接详情由 `/mcp`、`/memory` 查询。

## 6. 会话与结果持久化

首次普通消息才正式创建会话；帮助、空输入、退出不会产生空会话。会话JSONL首行为元信息，之后为事件；成功回合保存完整工具链与问答，压缩不删除磁盘原文。

- `/new` 准备新会话；`/sessions` 切换前保存当前会话，保存或加载失败保留原状态。
- `/rename 名称` 修改标题；内部ID不变。
- `/clear` 写入重置事件，清空当前上下文与活动结果索引，不删除之前磁盘原文。
- `/save` 手工保存或重试失败的保存。
- 恢复加载最近重置、摘要和回收边界后的有效投影；系统提示词按当前配置加载。
- 密钥不作为会话配置字段持久化；工具输出只对已知密钥做脱敏，不承诺全面敏感信息检测。
- 当前没有会话自动删除、历史自动擦除或多进程并发编辑同一会话的保障。

工具结果编号属于当前会话，`read_tool_result` 分页恢复历史快照，拒绝任意路径、跨会话ID和权限不符的来源。文件、网页、知识库、命令日志的恢复规则各自校验；读取快照不重新执行原工具。

## 7. 上下文预算、压缩与工具回收

本地窗口预算和供应商真实能力分开。当前项目显式配置如下，不意味着每次发送百万Token，也不是供应商规格承诺：

| 参数 | 当前项目值 |
| --- | ---: |
| `MODEL_CONTEXT_WINDOW_TOKENS` | 1,000,000 |
| `MODEL_MAX_TOKENS` | 8,192 |
| `CONTEXT_RESERVE_TOKENS` | 65,536 |
| `CONTEXT_KEEP_RECENT_TOKENS` | 131,072 |
| `CONTEXT_SUMMARY_MAX_TOKENS` | 8,192 |
| `CONTEXT_SUMMARY_GENERATION_MAX_TOKENS` | 32,768 |
| `CONTEXT_SUMMARY_INPUT_MAX_TOKENS` | 131,072 |
| `CONTEXT_SAFETY_TOKENS` | 32,768 |
| `TOOL_PRUNING_PRESSURE_RATIO` | 0.85 |
| `CONTEXT_COMPACTION_TRIGGER_RATIO` | 0.90 |
| `DEEPSEEK_TIMEOUT_SECONDS` / `RECOVERY_MAX_SECONDS` | 300 / 600秒 |

硬输入预算为934,464。每次请求测量完整消息、工具定义、推理元数据与运行快照；约794,294时尝试回收旧工具正文，剩余仍约841,017时尝试摘要旧回合。回收与摘要触发独立，不按固定轮数删除消息。

回收不是丢弃工具来源：按结构保留标题、URL、路径、行号和结果编号，正文缩为必要片段。近期工具批次、错误、未闭合调用和审批材料受到保护。完整结果通过当前权限下的快照或命令日志恢复。

摘要只覆盖已完成的历史，近期以完整回合保留；正在执行的工具链不摘要。旧历史分批合并，摘要生成预算与最终摘要目标分开。摘要校验并保存成功后才替换运行投影。失败时，若原请求仍在预算内则继续；无法容纳时明确暂停，不伪造摘要。

本地Token是启发式估算，API返回的 `prompt_tokens` 可用于向上校正低估并加安全余量。系数只在当前进程/会话有效，切换、清空、重启后重新采样。实际usage与估算分开显示。预算检查有缓存，不为缓存保留额外历史生命周期。

服务端明确上下文溢出时，只在缩小请求后有限恢复；不把一般网络或HTTP错误误判成窗口溢出。默认模型工具循环最多8轮、每次最多8个调用；工具即时结果默认16,000字符，不随1M窗口自动扩大。恢复总结可能只使用最近64条、每条4,000字符的有界资料投影，并标明截断。

当前百万Token配置经合成历史和模拟Provider测试；未以真实API发送百万Token请求验收。长文本手工测试资料在 `tests/fixtures/long_context/`。

## 8. 文件权限与审批档位

文件权限决定路径范围；审批档位决定询问频率。两者独立，完全权限不等于免审批，更不提升系统账户权限。

| 文件模式 | 边界 |
| --- | --- |
| `read` | 只读固定工作区，不允许读取工作区外，不提供写工具和Shell |
| `workspace` | 默认模式；工作区读写，覆盖仍需预览确认 |
| `full` | 扩展至当前账户可访问的允许路径，敏感、系统和内部保护规则仍生效 |

`/permissions full` 先提示风险，再 `/permissions full confirm`。敏感路径授权、审批与档位切换只由真实用户输入触发，模型文字不能批准自己。

| 操作 | `ask` | `auto`（当前默认） | `paused` |
| --- | --- | --- | --- |
| 只读 | 逐次卡片 | 自动 | 允许分析 |
| 新建文件 | 逐次卡片 | 自动 | 拒绝 |
| 修改已有文件 | 差异预览及确认 | 差异预览及确认 | 提交拒绝 |
| 批量替换、复制、改名、下载、解压 | 逐次卡片 | 可批准本批次 | 拒绝 |
| `exec_command` | 每条确认 | 每条确认，无批量允许 | 不可用 |

`/approval ask|auto|paused` 切换，`/approval revoke` 撤销批量授权，`/approval stop` 请求停止。Y允许一次，B允许本批次（仅支持的批量卡片），N/Esc拒绝；PgUp/PgDn浏览文件清单。Ctrl+X或停止按钮在文件边界停止，已完成的操作保留，不承诺自动回滚。

授权绑定会话、任务回合、文件权限、审批档位、操作类型、具体路径、文件集合、来源主机与端口、大小及操作上限；新增范围、切换会话/权限/档位、回合结束、撤销或重启使旧授权失效。拒绝的目标在当前回合内不能换另一写工具绕过。

应用拒绝UNC/设备路径、目录联接、符号链接、硬链接、上级路径、备用数据流等特殊路径。系统目录、内部会话/记忆/审批备份目录、配置及权限规则等受保护内容不能由通用文件工具任意改写。这些是应用校验，不能防御同账户程序或所有并发路径替换。

## 9. 文件与批量工具

普通工具包括 `list_dir`、`find_files`、`search_files`、`read_file`、`write_file`、`edit_file`。读取按行分页，常见文本与代码为主；不把二进制、PDF或Office当普通文本直接读取。已有文件替换需精确匹配并展示差异，审批后复查原内容哈希；新建不覆盖同名目标。现有覆盖审批不等于所有文件修改已实现回合内自动闭环。

| 批量工具 | 当前限制 |
| --- | --- |
| `batch_replace_text` | 明确文件列表、UTF-8固定字符串替换，单文件≤1MiB；覆盖前备份与哈希复查 |
| `batch_copy_files` | 只新建、不覆盖；来源变化则跳过并清理不完整目标 |
| `batch_rename_files` | 同目录改名，不覆盖；记录新旧路径供手工恢复 |
| `download_file` | 公网HTTP/HTTPS、来源审批、大小/超时/重定向限制，不覆盖、不自动执行 |
| `extract_archive` | ZIP/TAR/TAR.GZ/TGZ；拒绝路径越界、链接、设备、重名、加密ZIP和超限 |

下载不发送Cookie/Authorization、不读取代理环境变量；每次重定向重新检查来源，不能绕过公网访问规则。下载与解压不等于执行。解压限制成员数、总大小和压缩比，不支持7z/RAR。

审计位于运行数据 `approval/audit.jsonl`，备份位于 `approval/backups/`，不写文件正文或已知密钥。它们不是防篡改账本；审计故障不保证阻止操作。已有修改、复制、改名、下载结果需按记录人工恢复，没有通用自动回滚或永久删除工具。

## 10. 本机命令执行

交互式TUI/逐行界面启动时当前默认开启Shell，但每条命令仍须人工批准；`/shell off`关闭，`/shell status`查看。再次显式开启会提示风险，使用 `/shell on confirm`。headless和没有操作员审批界面的入口不可用；暂停执行或仅查看模式也不可用。

`exec_command` 接受 `argv` 数组、`working_dir`、`timeout_seconds`。程序先解析为绝对路径，直接启动，不自动拼接Shell字符串，不支持管道、重定向、交互输入或后台常驻。默认60秒、上限300秒，审批等待最多10分钟，不受普通工具30秒超时截断。

卡片绑定实际程序、完整参数、目录、超时、会话、权限与直接脚本SHA-256；修改后原审批作废。导入模块、运行期下载与动态配置不在直接脚本哈希保护范围。Shell没有“允许本批次”。

stdout/stderr持续读取，已知密钥脱敏后保存完整日志；模型只获得有限头尾、退出码、耗时、总字节数、截断标志与 `log_result_id`。非零退出码是程序运行结果，不直接当成工具状态未知。启动失败、超时、取消和读取异常要如实标明；不得自动重试可能已产生副作用的命令。

Windows使用Job Object清理进程树，不可用时回退taskkill；POSIX使用进程组。日志按会话保存在 `execution/<会话UUID>/<结果UUID>.log`，普通文件工具不可访问；`read_tool_result`按当前会话编号分页读取。清空上下文不删除旧日志。`/retry`只整理已有结果，不重跑命令。

**不是沙箱。** 工作目录只决定从哪里启动；获批程序仍可读取`.env`、访问工作区外、联网或修改当前账户能操作的系统资源。环境白名单、审批和脱敏不能阻止这些行为。当前允许必要系统变量、非密钥UV变量和代理变量；脱敏只能覆盖已知配置密钥，不能检测全部秘密或编码后的密钥。

## 11. 联网搜索与网页读取

`web_search` 默认AnySearch后端，也支持已配置Brave；`/search off`关闭关键词搜索，`/search on`恢复。网页读取独立，不得在搜索关闭后拼搜索URL绕过。AnySearch的时间范围提示不保证服务端严格过滤，最近事实需核对原文日期。

`web_fetch`读取已知公开URL正文，不执行网页脚本、不绕过登录/验证码，不伪造访问成功。检索摘要不是原文；回答附来源，区分事件日期、发布日期与检索日期。请求有超时、响应大小和重定向限制，遵循公网URL安全检查。

模型不能自行获取实时信息，必须实际调用工具。测试入口 `tests/manual/check_web.py` 不调用模型，但可能消耗搜索服务额度；不能把后端已配置等同于网络连通。

## 12. MCP、长期记忆与档案

MCP目前支持本地stdio与仅本机IP、认证的Streamable HTTP工具连接。服务必须在配置中显式启用，并限定工具白名单；模型不能自动下载服务或改配置。`/mcp`查看状态，`/mcp tools`查看工具，`connect/disconnect 服务名`管理已配置连接。HTTP空闲保活使用工具发现，不使用该服务不支持的旧ping。

Engram 3.4.0以本地stdio MCP连接，数据统一保存在本地记忆目录。SOUL代表风格、USER代表稳定偏好、AGENTS代表项目长期规则，确定性加载；普通经验按相关性检索。不另外维护同名Markdown文件，不默认导入旧聊天或推断用户画像。

- `/memory list/search/show`查看记录，`/memory changes`查看变更，`/memory autosave on|off`、`recall on|off`管理自动行为。
- `/profile show soul|user|agents`查看档案；初始化和修改只提出候选，需用户预览批准。
- 成功回合先保存Session，再有限提取少量重要结论；不是每句话都存，失败或截断回合不伪装成成功结论。
- 普通重要结论可自动追加，档案变更、人工提案、覆盖和删除仍遵守审批流程；没有后台Dream式自动合并删除。
- 写入保持类型、项目、作用域、来源与版本；自动召回是历史参考，不是当前事实或更高优先级指令。
- 新会话可以召回同作用域记忆，但不能混用其他项目；原始聊天由Session保存，不复制为长期记忆。

配置启用不保证Engram进程正在连接；使用 `/memory` 或 `tests/manual/check_memory.py`检查，后者不调用模型但首次连接可能创建运行元数据。

## 13. 文档RAG与知识库

文档知识库不是Engram。当前已授权知识库标识为 `deepseek_docs`，包含DeepSeek工具调用摘要和保留的会话测试文档，不是官方完整文档或全项目资料。

独立服务通过本机HTTP MCP只开放 `search_knowledge(query, kb_id, top_k)`；知识库必须明确指定并独立授权，不扫描磁盘。即使文件工具为完全权限，也不自动扩大知识库范围。

RAG使用固定版本BGE-M3，dense+sparse检索与RRF融合；Milvus保存向量，CPU单次推理、有界队列。默认600 Token切块、80重叠，来源保留文件、章节、页码（有则返回）、chunk_id、index_version。支持TXT、Markdown、文本PDF、显式字段映射JSON/JSONL，不支持OCR、图片和任意URL自动导入。

离线构建仅处理显式导入清单；文件失败需修复或明确排除，不能静默漏文档。新物理版本验证成功才发布alias，失败保留旧库。更新/删除文件需全量重建，不实时同步。查询时固定物理版本，并在召回前后检查项目、库和版本。

### 本机部署与更新

正式部署继续使用独立 `lynx-rag-validation` Compose项目已有容器和数据卷（名字保留历史命名），数据库端口19540；不要操作课程的19530实例。服务地址为 `http://127.0.0.1:8766/mcp`。

运行配置在 `E:\github\lynx_agent_data\rag\` 的 `settings.toml`、`grants.json`、`import.json`；认证从受保护的 `mcp_token.key`文件读取，不把值放进命令参数。Agent配置中 `token_file` 与 `token_env`只能选一个，当前使用前者。

导入清单中已把会话测试文档路径更新到 `tests/fixtures/document_rag/session.md`。目录整理不改旧索引与历史会话，旧来源路径代表建库时版本；下次重建会使用新路径和新版本。

更新顺序：停止查询服务 → 编辑显式清单 → 建库成功 → 重启服务与Agent。同一state目录的查询和建库互斥，不要手工删除锁文件，不要在服务仍运行时建库。

```powershell
cd E:\github\nanobot\services\document_rag
uv run --frozen rag-build --config E:\github\lynx_agent_data\rag\settings.toml build --plan E:\github\lynx_agent_data\rag\import.json --kb deepseek_docs
cd E:\github\nanobot
uv run python scripts\rag_runtime.py agent
```

只停止使用该配置的RAG进程，不停止其他项目或数据库。退出Agent不等于停止后台RAG。不要执行 `docker compose down -v` 删除数据卷。组件安装、回滚和手动清理的命令入口见 [RAG组件说明](../services/document_rag/README.md)。

知识库更新保留旧结论的来源版本。撤销授权阻止新查询、工具快照恢复和关联记忆再次自动召回；不会自动擦除旧会话正文、摘要或已经发给模型的内容。RAG结论自动入记忆时，来源由程序绑定，不能接受模型编造的source/version。来源不可验证或超出限额时跳过保存。

## 14. 故障恢复、审计与取消

- 摘要失败：原请求放得下时保留历史继续；超预算时暂停，不伪造摘要或原样无限重试。
- 模型截断：有限次数精简重写；仍不完整则保存草稿，不加入成功上下文。
- 空白推理耗尽预算：记录诊断，等待用户调整，不无意义自动追加费用。
- 工具超时：返回实际失败；写入和命令可能已部分发生，必须核实，不能盲目重做。
- 网络失败：与认证、限流、格式错误区分；不把失败当“查无资料”。
- 保存失败：停止后续请求，安全进度尽量留在内存；先 `/save`，再 `/retry`。
- `/retry`：仅根据已有资料生成最终答复，不提供新工具权限，不重复执行已经发生的命令。
- 工具批次未闭合、取消或崩溃不一定有安全恢复点；不承诺断电时完整恢复正在执行的任务。

`/diagnostics`提供阶段、异常类别、恢复动作、耗时和生成预算，不泄露底层密钥与原始推理。工具审计、Session与运行日志分工，不把UI状态误认为模型消息。

## 15. Token、缓存与费用

`/tokens`同时区分本地完整请求估算与API实际usage。工具定义、历史、系统提示与运行资料都计入预算；缓存命中率由返回的缓存输入Token加权计算，缺失字段不当成零。

系统固定规则和工具排序保持稳定，动态时间、权限和召回资料在任务边界采集，并在本轮工具循环中复用，降低无谓前缀变化。但不能承诺服务端每轮缓存命中率。

费用按当前程序内实际返回的usage、模型与请求发起时刻的计价配置估算，包括普通对话、摘要、重写、结论提取；不等于账户账单。取消、缺失usage、缓存拆分或未知模型时标记统计不完整。峰谷提示依据代码计价规则与适用日历，不自动推断未知年份或第三方代理价格。搜索/MCP服务费用不计入模型费用。

## 16. 调试、测试与评测入口

测试与人工检查统一放在 `tests/`，正式启动器继续留在 `scripts/`。不把演示运行等同于真实模型质量验证。

```powershell
cd E:\github\nanobot
uv run --locked pytest -q
uv run --locked ruff check src tests
uv run python tests\manual\demo_compaction.py
uv run python tests\manual\tui_preview.py
uv run python tests\manual\demo_tui.py --tools
```

- 人工命令、是否联网、是否可能收费见 [测试入口](../tests/README.md)。
- 长期记忆、审批、RAG测试方案在 `tests/plans/`。
- 长上下文6段文本在 `tests/fixtures/long_context/`，按顺序手工粘贴，不是系统提示词。
- 三份RAG测试文档与验证配置在 `tests/fixtures/document_rag/`；独立验证库项目是 `rag_validation`，与正式项目授权分开。
- 压力测试需显式设置 `LYNX_PRESSURE_TEST=1`，普通测试不自动开启重型进程负载。
- 真实RAG联调需 `LYNX_RAG_REAL=1` 与对应组件就绪；默认skip，不能把未启动数据库当测试成功。
- 保留历史报告在 `tests/reports/`，其中路径、版本与测试数是当时快照，不代表当前结果。

独立评测包 [agent-bench](../bench/README.md)使用套件、隔离工作区、录制代理、确定性评分与报告。`bench validate`检查用例；真实 `bench run`会调用配置的模型、消耗费用。`results/`保留评测产物，不当源码或知识库自动导入。

## 17. 已知限制与后续边界

### 训练工程助手 M1（新增基础能力）

已实现 Windows→WSL 的短请求桥、环境/GPU只读探测、用户显式注册项目、静态发现SFT/GRPO候选入口、本地SQLite人工实验台账、有限数值差计算与接入计划。默认关闭，`/training on confirm`在当前运行期开启；不会修改本地配置，不依赖模型API完成这些命令。

`/training environment`查看环境；`projects`列已注册项目；`register <WSL源码绝对路径> confirm`授权该目录用于静态发现；`inspect <项目ID>`查看候选与证据；`plan <项目ID>`给出不可执行的接入计划；`records <项目ID>`查询人工台账；`record <项目ID> <sft/grpo/other> <协议名> <指标=数值>... confirm`登记；`compare <项目ID> <基线实验ID> <候选实验ID>`仅在协议名相同时计算共有指标差值。路径有空格需引号；M1注册暂只接受Linux绝对路径，不猜测Windows路径映射。

模型可用training_inspect/status/records/plan/compare五个只读工具；不能自行注册目录、保存指标、提交或控制任务。项目注册与人工登记写入要求文件权限可写、审批不为paused；真实终端的confirm命令是本阶段批准入口。读取仅限显式登记项目，未注册目录不会被扫描。

新增训练只读观察器：`/training progress <项目ID>`采集项目Python进程及最近历史日志/指标；`files <项目ID> [偏移]`分页列产物元信息；`logs <项目ID> <相对路径> [距末尾字节偏移]`读取日志；`metrics <项目ID> <相对路径>`读取数值和完成标志。对应新增training_progress/logs/metrics/artifacts四个只读模型工具，总计九个。DeepSeek可总结采集结果，斜杠命令本身无需模型API。

一次注册授权后，只读监测不受通用工作区路径或可写模式限制，不逐次请求Shell批准；不因此允许读任意文件、跨项目、修改产物或启停训练。仅项目内有界日志/指标读取，权重只列元信息，拒绝隐藏路径、数据集及链接。采集时间、文件时间和来源保留，进程探测与历史文件分开；无进程证据不判定running，历史完成/错误也不冒充当前状态。日志分页和扫描可能截断；无法获取的阶段/步数明确未知，不从GPU占用猜测。按需查询，不是持续监控/自动告警，不能直接抓取其他PowerShell的历史屏幕。操作说明见[worker README](../services/training_worker/README.md)。

默认台账为WSL的`~/lynx-training/registry/experiments.sqlite3`，只有确认注册/登记才建库，环境与空列表查询不创建运行数据。人工记录始终标记user_reported_unverified，缺少代码、数据和训练证据，不能用于证明训练提升。数值差不等于公平实验对比。关闭模块不删除数据库。

M1使用标准库bootstrap短进程，SQLite短事务串行化写入；尚未部署单写者服务、持久调度、训练审批凭据、框架暂停续训、自动改训练项目代码或完整受控比较。普通Shell限额和审批规则不变。调用路径依赖此源码仓库，当前不是可独立分发的worker安装包。接口与目标架构见[技术设计](training-assistant-technical-design.md)，复测与后续训练步骤见[接入计划](../tests/plans/training-onboarding.md)。

### 其他边界

- 没有操作系统沙箱、独立账户强制隔离或默认网络隔离；Docker目前承载RAG数据库，不隔离本机Shell。
- 没有持久交互式Shell、通用浏览器自动化、无人值守安装/发布、FastAPI前后端或多用户服务。
- 文件修改仍保留既有差异审批，完整任务闭环允许人工介入，不能宣称全部自动完成。
- 自动测试和模拟Provider不能证明真实模型工具选择、摘要质量、记忆准确率或大语料检索效果。
- Markdown文档中的架构说明、测试资料、运行时提示词作用不同；不要按扩展名批量删除。
- 缓存、`.venv`、`.uv-cache`、IDE设置与`results/`本次保持；运行数据、旧索引与记忆数据库没有清理。
- 凭据仍需在Git提交前单独检查，当前整理不替用户上传、推送或提交任何内容。
