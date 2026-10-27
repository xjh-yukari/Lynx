# Lynx 文档检索 MCP

独立 Python 3.11 环境、BGE-M3、Milvus Standalone、dense+sparse+RRF(k=60)。不增加第二个 Agent，不修改 Engram，不执行 all-in-rag 示例。在线只开放 `search_knowledge`。

## 目录与职责

- `config.py`：配置、编码清单、独立授权。
- `encoding.py`：离线/在线共用 BGE-M3 编码器，1024维、归一化 IP、CPU fp32，不生成 ColBERT。
- `documents.py`：TXT/MD/文本PDF/JSON/JSONL、真实页码、结构切块、稳定主键。
- `storage.py`：单一 MilvusClient、物理版本固定、服务端过滤、发布alias、进程互斥。
- `build.py` / `cli.py`：离线批量建库、验证后发布、回滚、显式旧库清理。
- `retrieval.py` / `server.py`：单实例编码器、有界队列、只读HTTP MCP、认证及Host/Origin限制。
- `../../tests/fixtures/document_rag/` / `../../tests/manual/validate_rag_real.py`：隔离测试语料和真实组件验证；不是用户正式知识库。统一入口见 [测试说明](../../tests/README.md)，产品能力见 [产品方案](../../docs/product-plan.md)。

## 安装

在此目录执行：

```powershell
uv sync --frozen
```

此目录有自己的 `.venv` 和 `uv.lock`，不要在 Agent 根环境安装 RAG 重依赖。CPU Torch 使用官方CPU wheel索引。版本按锁文件安装，不执行课程 requirements。

切块按tokenizer字符偏移截取原文，不通过decode重写空格和换行。数据库、模型和中间清单为本地明文数据，应由操作系统控制目录权限。默认本机Milvus端口不是面向恶意本机进程的安全沙箱；多用户部署应另配数据库只读账户与网络隔离，首版不开放远程部署。

模型固定 commit 为 `5617a9f61b028005a4858fdac845db406aefb181`，只下载所需权重、tokenizer和配置。离线使用时将 `model` 设为同一快照的绝对路径；离线建库与在线必须使用完全相同的配置和模型标识。首次初始化需要较多内存，CPU查询不是即时返回。

## 启动数据库

先为此功能设置独立 MinIO 凭据，勿使用课程的数据库卷或容器：

```powershell
$env:LYNX_MINIO_USER = 'lynxrag'
$env:LYNX_MINIO_PASSWORD = [guid]::NewGuid().ToString('N')
docker compose -p lynx-document-rag -f compose.yml up -d --wait
```

请自行安全保存凭据，以便后续用同样环境维护该Compose项目。默认19540和19091端口仅发布在127.0.0.1，不占用课程的19530。MinIO/etcd没有宿主机发布端口，使用独立命名卷。**不要运行 `down -v`，它会删除知识库数据卷。** 在线服务不会创建、release或删除collection。

当前开发验证另用了 `lynx-rag-validation` 项目；若它仍占用19540，应先停止验证容器，再启动正式项目。不要停止名为 `milvus-standalone` 的课程实例。

## 配置与显式授权

1. 参照 `settings.example.toml` 创建运行配置。`state_dir` 和授权文件建议放在 `E:\github\lynx_agent_data\rag`。
2. 参照 `grants.example.json` 创建授权文件；Agent与服务读取**同一文件**。`project_id` 必须与 Agent 当前项目身份相同，不能直接照抄示例。
3. 运行以下只读命令查看实际项目ID（不启动模型或Engram）：

```powershell
cd E:\github\nanobot
uv run python -c "from lynx_agent.config import load_config; from lynx_agent.agent import AgentRunner; from lynx_agent.providers import create_provider; c=load_config(); r=AgentRunner(c,create_provider(c)); print(r.memory.project_id)"
```

4. 只把你授权的文件写进导入计划JSON。路径必须为绝对文件路径，不扫描目录。JSON字段映射示例见 `import.example.json`。`asset` 仅保存图片路径元数据，不读取图像。

不得将密钥写在Agent对话或MCP工具参数中。授权文件可用原子替换方式将某个grant设为 `enabled:false`；每次检索、结果返回和记忆读取均重新检查。删除grant同样撤权。

授权文件由用户在Agent之外修改；它被加入文件工具的保护范围，即使“完全权限”也不能让模型自行改grant扩大读取范围。部署时不要另开能绕过该限制的shell/MCP写工具。

## 离线建库、发布、更新与回滚

先停止使用同一state目录的在线服务。文件锁禁止一边在线占用模型、一边离线建库；首版不承诺不停机重建。构建本身仍采用新物理collection，失败不会改旧alias。

```powershell
uv run --frozen rag-build --config E:\github\lynx_agent_data\rag\settings.toml build --plan E:\github\lynx_agent_data\rag\import.json --kb default
```

失败会写 `state_dir/last_build_failure.json`。修复文件或**明确排除**后重试：

```powershell
uv run --frozen rag-build --config E:\github\lynx_agent_data\rag\settings.toml build --plan E:\github\lynx_agent_data\rag\import.json --exclude E:\资料\扫描件.pdf
```

排除后仍须有可用正文，不能发布空库。未排除的任何文件失败都会禁止发布。不支持扫描PDF/OCR；有空白/无正文PDF页时整份文件报错，不静默漏页。每个新库保存文档哈希、配置、排除清单和片段数。每批分别编码和插入；计数、加载、查询验证成功才切alias。重复输入和配置识别同一逻辑版本。更新或删除源文件，需修改计划后重新全量构建；不做实时同步。

回滚使用构建输出里的**真实**物理collection名，不填写尖括号占位符：

```powershell
uv run --frozen rag-build --config E:\github\lynx_agent_data\rag\settings.toml rollback --kb default --collection 真实旧版本名称
```

`publish` 命令同样可重新发布一个已验证版本。`cleanup --collection 真实旧版本名称` 仅允许删除无alias、清单归属正确的旧版本。运行前停止所有读取该库的服务；仅共享本state目录的进程受文件锁保护，其他管理员客户端不能据此视为安全。

## 在线查询与进程管理

在服务和启动Agent的PowerShell中设置同一随机密钥：

```powershell
$env:RAG_MCP_TOKEN = '你安全生成并保存的至少32字符随机密钥'
uv run --frozen rag-serve --config E:\github\lynx_agent_data\rag\settings.toml
```

服务在模型、数据库版本检查及预热完成后才监听 `127.0.0.1:8766/mcp`，不能通过新MCP会话重新加载模型。Ctrl+C关闭；不删除库或卸载其他客户端的collection。

需要长期运行时，可在Windows“任务计划程序”创建你自己的任务：程序选择此服务 `.venv\Scripts\rag-serve.exe`，参数为 `--config 你的配置绝对路径`，起始目录为此服务目录；使用低权限账户，为其安全注入环境变量，设置失败后延迟重启。不要把密钥放在启动参数，不启用多个副本。首版不替你注册系统任务或写全局环境变量。

## Agent接入

仅将以下配置**合并**进 `E:\github\nanobot\config.toml`，不要覆盖既有模型和memory配置：

```toml
[knowledge]
enabled = true
policy_file = 'E:\github\lynx_agent_data\rag\grants.json'
server_name = "documents"

[mcp.servers.documents]
enabled = true
transport = "streamable_http"
url = "http://127.0.0.1:8766/mcp"
token_env = "RAG_MCP_TOKEN"
allowed_tools = ["search_knowledge"]
read_only_tools = ["search_knowledge"]
allow_write = false
startup_timeout = 30
call_timeout = 120
```

重启Agent，`/mcp`检查连接，`/tools`检查 `mcp__documents__search_knowledge`。使用token_env时，密钥必须在**启动Agent的进程环境**中；仅放在项目 `.env` 不会自动导出给HTTP客户端。工具还有Agent全局超时，需按CPU延迟设置现有 `TOOLS_TIMEOUT_SECONDS`，否则它可能早于MCP超时生效。

也可把示例中的 `token_env` 替换为 `token_file = 'E:\github\lynx_agent_data\rag\mcp_token.key'`，由Agent直接读取受保护的本地密钥文件，不依赖终端环境继承。二者只能选择一个。当前本机部署已采用token_file，详见 [产品方案](../../docs/product-plan.md)。工具列表也可使用 `/mcp tools`。

HTTP空闲保活使用只读工具发现，每15秒检查一次；不调用本HTTP服务不支持的旧ping。stdio仍使用原ping方式。短连接探测通过不代表长期连接已验证，复测必须跨过保活周期后再查询。

## 记忆与权限边界

Engram保持原部署、schema和数据；只在Lynx自己的记忆协议中增加可选来源字段，旧记录序列化不变。自动重要结论仍可保存，但由程序绑定参考上下文里的知识库、项目、索引版本、chunk及来源。模型不能自由指定这些元数据。采用保守来源集合，不声称是逐句引用对齐；超过32个来源或来源不可核验时跳过保存，不能去掉来源降级写入。

更新后旧结论继续保留并标注历史版本；撤权后禁止新查询、旧工具快照恢复及关联记忆再次读取/自动召回。**不抹除旧会话正文、已发送给模型的内容或已有摘要**，它们仍可能在恢复会话时再次发送。通用工具不能绕过此授权去读取新资料。文件权限与知识库授权独立。

RAG工具返回内容是不可信参考资料。知识库查询不直接触发写记忆；任务完成后的既有自动结论流程可能写入带来源记忆。SOUL/USER/AGENTS仍需人工批准，不自动写入。RAG MCP没有建库、写文件或记忆写工具。

## 测试与限制

Agent根目录：`uv run pytest tests/knowledge -q`。覆盖授权撤销、旧记忆兼容、真实HTTP MCP及真实临时Engram来源绑定；不调用付费模型。

此目录：`.venv\Scripts\python.exe ..\..\tests\manual\validate_rag_real.py`。要求验证Compose在19540就绪；使用固定三份测试语料、真实BGE-M3、真实Milvus、真实HTTP客户端。输出 `../../tests/reports/rag-last-real-report.json`。该报告不等于真实LLM路由或用户正式知识库的验收。

三题 Recall@5 只是冒烟评测，不能证明混合优于dense；引用元数据正确不等于最终回答引用正确。不引入视觉检索、cross-encoder、网页UI、Redis或新记忆系统。

进一步的真实本地联调（模型路由用脚本替身，**不是付费DeepSeek实测**）：验证库就绪后，在Agent根目录设置 `$env:LYNX_RAG_REAL='1'`，执行 `uv run pytest tests/knowledge/test_full_local.py -q`。它启动真实RAG HTTP服务、复用真实BGE/Milvus、使用临时Engram库，验证四种工具路径、来源写入与撤权。未设置变量时明确skip，普通测试不隐式启动重服务。

容器重启后可从本目录运行 `.venv\Scripts\python.exe ..\..\tests\manual\check_rag_persistence.py`，检查alias、数据和加载状态；该脚本不会创建或load库。大语料评测、真实LLM工具选择准确性和最终回答引用正确性仍需后续验收。

参考：[Milvus 2.5 hybrid_search](https://milvus.io/api-reference/pymilvus/v2.5.x/MilvusClient/Vector/hybrid_search.md)、[BGEM3EmbeddingFunction](https://milvus.io/api-reference/pymilvus/v2.5.x/EmbeddingModels/BGEM3EmbeddingFunction/BGEM3EmbeddingFunction.md)、[MCP客户端](https://py.sdk.modelcontextprotocol.io/client/)。
