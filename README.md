# Lynx Agent ｜轻量化个人智能助手

面向个人开发与资料处理的终端智能助手：流式对话、受控工具调用、可恢复会话、长期记忆，以及可选的文档检索与训练观察器。

当前 Python 包版本为 **V1.0.0**。`V1.0.0` 是历史基线标签，后续改动以 Git commit 为准。项目仍在演进，不是无人值守生产平台。

## 能做什么

- **终端对话**：TUI、多行输入、Markdown、复制、滚动、流式输出和取消。
- **工具执行**：文件查询与受控修改、逐条审批的本机命令、可选联网搜索及网页读取。
- **会话与上下文**：JSONL 原文保存、恢复、切换；Token 估算、摘要压缩、旧工具结果回收。
- **长期记忆**：可选 Engram MCP；管理用户偏好、项目规则和可检索的重要结论。
- **文档 RAG**：可选独立服务，使用 BGE-M3、Milvus 和认证 HTTP MCP，保留来源与索引版本。
- **训练观察器**：已授权 WSL 项目的进程、日志、步数和指标只读采集，交给模型总结。**尚不支持训练提交、暂停或完整续训。**

“轻量化”指核心 Agent 与重型组件分离：基础对话不需要 Docker、Torch 或 GPU；RAG 有自己的环境及额外资源成本。模型推理由 API 完成，不是本地大模型推理。

## 快速开始：先运行基础对话

主要验证环境为 Windows + PowerShell，Python **3.11+**、uv。其他平台未做同等完整验收；训练桥当前专门面向 Windows → WSL2。

在下载或克隆的源码根目录执行：

```powershell
uv sync --frozen
Copy-Item .env.example .env
```

如果已有 `.env`，**不要覆盖**。自行编辑其中的 `DEEPSEEK_API_KEY` 和可用的模型名称，然后启动：

```powershell
uv run --locked lynx-agent
```

这是基础入口，**不需要启动 Docker/RAG/Engram**。示例默认关闭联网搜索；按 [web.example.env](web.example.env) 和 [config.example.toml](config.example.toml) 选择启用其他能力，不要直接复制他人的本机配置。

DeepSeek 客户端目前固定忽略环境代理、直连配置的 API 地址，保留 HTTPS 证书校验；其他客户端保持各自网络设置。使用自定义 `DEEPSEEK_BASE_URL` 时，请先确认你信任该目标，密钥与请求内容会发送到那里。系统 TUN/透明代理不由此设置控制。

## 常用操作

| 操作 | 命令 |
| --- | --- |
| 帮助与快捷键 | `/help` |
| 新会话 / 选择会话 | `/new` / `/sessions` |
| 查看上下文 / 费用估算 | `/tokens` / `/cost` |
| 最近故障 | `/diagnostics` |
| 长期记忆状态 | `/memory` |
| 训练只读观察器 | `/training on confirm`，然后 `/training projects` |

训练项目必须先由用户显式注册。已经授权的项目可使用 `/training progress <项目ID>`，也可直接让模型总结训练阶段、最新步数和异常。历史结果不等于当前进程状态。

## 可选组件

| 组件 | 安装与边界 |
| --- | --- |
| Engram / MCP | [产品方案](docs/product-plan.md)，需自行安装并配置 Engram |
| 文档检索 | [RAG 服务说明](services/document_rag/README.md)，独立环境、Docker 数据库、显式导入与授权 |
| 训练观察器 | [worker 说明](services/training_worker/README.md)，需已配置 WSL，当前仅观察与人工台账 |
| 独立评测 | [agent-bench](bench/README.md)，真实模型评测可能产生费用 |

`scripts/rag_runtime.py` 和开发机的 `lynx` 快捷命令是已有部署的辅助入口，不是通用首次安装器。新用户请先使用上述基础入口；可选组件文档中的绝对路径、项目身份和运行目录必须按本机调整。

## 数据与安全

源码安装模式下，`.env` / 可选的 `config.toml` 位于源码根目录，运行数据默认放在仓库同级的 `lynx_agent_data/`，不自动成为版本库内容。

**审批不是操作系统沙箱。** 获批命令仍具有当前账户的文件和网络权限。文件、日志、检索片段与历史记忆可能被发送给模型服务；不要在不受信任的目录随意开启执行能力，也不要上传真实密钥、会话、记忆数据库、实验数据或私人日志。

发布前运行 [安全检查](docs/open-source-release.md)。`.gitignore` 不能清除已经存在于 Git 历史中的敏感信息。漏洞和泄密处理见 [SECURITY.md](SECURITY.md)。

## 开发与贡献

```powershell
uv run --locked pytest -q
uv run --locked ruff check src tests services/training_worker
```

普通测试不调用真实模型。可选集成和压力测试的环境、费用及副作用见 [测试说明](tests/README.md)。贡献流程见 [CONTRIBUTING.md](CONTRIBUTING.md)，完整能力边界见 [产品方案](docs/product-plan.md)。

## 致谢与许可证

本项目在开发中参考了 nanobot、pi-mono、dsh-TUI 等项目的设计，并可集成 Engram；不属于这些项目的官方发行版。详见 [来源与第三方说明](THIRD_PARTY_NOTICES.md)。

本项目采用 [Apache-2.0](LICENSE)，版权说明见 [NOTICE](NOTICE)。第三方代码、依赖和素材仍遵循各自许可证。本次仅准备公开文件，尚未创建或推送 GitHub 远程仓库。
