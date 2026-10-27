# Lynx Agent

当前版本：**V1.0.0**（Python 包版本 `1.0.0`，Git 标签 `V1.0.0`）。

本地单用户终端 Agent：DeepSeek 对话与工具循环、受控文件/命令执行、Engram 长期记忆、文档RAG、上下文管理与独立评测。

## 产品与安全边界

功能、架构、配置、审批、运行路径与已知限制统一维护在 [产品方案](docs/product-plan.md)。

当前本机命令执行不是沙箱。审批不限制获批程序访问当前系统用户的文件与网络。密钥、会话和记忆运行数据不要提交到仓库。

## 启动

本机已有环境时，先打开Docker Desktop、等待引擎就绪，然后输入：

```powershell
lynx
```

通用入口：

```powershell
cd E:\github\nanobot
uv sync --frozen
uv run python scripts\rag_runtime.py agent
```

模型配置来自本地 `.env` 与 `config.toml`；示例见 `config.example.toml`、`web.example.env`。RAG在独立环境安装，不能直接用本启动命令替代首次部署。快捷命令固定使用现有工作区，不自动信任调用者所在目录。

## 维护入口

- [协作、测试与提交规则](AGENTS.md)：每个独立改动配套测试，验证通过后创建提交。
- [测试、人工调试与测试资料](tests/README.md)
- [独立RAG组件](services/document_rag/README.md)
- [agent-bench评测包](bench/README.md)
- [训练工程助手需求讨论稿（尚未实现）](docs/training-assistant-requirements.md)

```powershell
uv run --locked pytest -q
uv run --locked ruff check src tests
```

源码在 `src/`；调试程序在 `tests/manual/`；固定语料在 `tests/fixtures/`。运行数据保存在仓库外的 `E:\github\lynx_agent_data`。缓存、环境和 `results/` 不因目录整理被清理。
