# Lynx 训练工程助手技术设计

文档版本：0.1。日期：2026-10-27。需求基线：[训练工程助手需求 v0.3](training-assistant-requirements.md)。

实施状态：技术设计稿，尚未实现。示例接口、配置、SQL 和命令均为待实现契约，不是当前可执行功能或部署指令。本文不授权安装服务、启动训练、迁移数据或修改正在运行的实验；不移动 V1.0.0 标签。

## 1. 设计目标与关键决策

实现“用户需求 → 设计与代码 → 测试 → commit → WSL 训练 → 评测 → 历史对比 → SQLite 记录”，由程序保留事实，DeepSeek 做规划、代码生成和解释。目标不包含新训练算法、云端调度、远程 SSH、多用户或新的论文贡献。

| 编号 | 技术决策 | 原因 |
| --- | --- | --- |
| D01 | Windows Agent + WSL 独立 worker | 复用现有交互；训练与模型/TUI生命周期分开 |
| D02 | wsl.exe --exec 客户端 + 本地 Unix socket | 不增加公网 HTTP 服务、Redis 或新训练 MCP |
| D03 | WSL SQLite 单写入方 + 事务 outbox | 持久化、幂等、崩溃对账和可恢复派发 |
| D04 | systemd 用户级任务托管 | 使用受管 unit/进程组，不把训练塞进普通 Shell 工具 |
| D05 | 不可变计划 + 独立训练审批凭据 | 现有回合级授权不适用于跨会话任务 |
| D06 | 固定代码快照、验证后提交 | 运行任务不受之后代码修改影响 |
| D07 | checkpoint 完成确认后暂停退出 | 不把 SIGSTOP/SIGTERM 或只存权重当完整续训 |
| D08 | GPU 密集任务串行，运行时限可为 null | 符合本机资源与持续运行需求，不自动无限试验 |
| D09 | 框架适配器按证据加载 | 源码未提供前不预设实际使用 TRL、Unsloth 或其他框架 |
| D10 | 训练事实与 MCP/Engram 经验分离 | 记忆/API故障不阻塞训练及事实记录 |

本机探测快照见需求第 10 节：Ubuntu-24.04/WSL2、RTX 5070 约12GB显存、Windows约32GB内存、WSL约15.5GiB内存。该快照不保证任意模型可训练；提交时重新检测。训练源码位置尚未提供，实际框架与模型兼容性未验证。

## 2. 部署与信任边界

```text
Windows
  用户 → TUI / ApprovalPresenter
             ├─ DeepSeek Provider：规划、代码、解释
             ├─ TrainingTools → TrainingClient
             └─ TrainingApprovalBroker（仅真实用户决策）
                         │ wsl.exe --distribution Ubuntu-24.04 --exec
WSL                      ↓ stdin/stdout JSON，单次短请求
  worker client → Unix socket → TrainingService
                                  ├─ RegistryWriter → SQLite
                                  ├─ Dispatcher / Reconciler → systemd 用户任务
                                  ├─ ProjectPolicy / Git / Workspace
                                  ├─ AdapterRegistry → 项目解释器 → 训练/评测
                                  └─ Logs / Metrics / Checkpoints / Reports
```

控制客户端不执行长训练；返回任务编号后即可退出。worker 不调用 DeepSeek，不持有模型 API 密钥，训练进程使用自己的已批准凭据引用。只读查询不触发 GPU 分配、下载或隐式任务提交。

### 2.1 与现有源码的集成

| 当前入口 | 现状 | 待新增集成 |
| --- | --- | --- |
| src/lynx_agent/tools/base.py | Tool.execute 返回字符串 | TrainingTool 返回有界 JSON 字符串，不返回巨量日志 |
| src/lynx_agent/tools/registry.py | 按可用性过滤并稳定排序 | 配置启用且项目授权有效时注册固定训练工具组 |
| src/lynx_agent/agent/runner.py | 构建文件、执行、记忆等服务 | 创建 TrainingClient/ServiceFacade，关闭时仅断开客户端，不停止任务 |
| src/lynx_agent/agent/tool_executor.py | 一般工具受超时、写保护和重复调用限制 | 为训练操作增加显式受控类型与状态映射，不能全局放开写工具 |
| src/lynx_agent/tools/builtin/shell.py | exec_command 为短命令，最大300秒 | 保持原限额，不增加后台常驻或任意管道能力 |
| src/lynx_agent/approval/manager.py | 授权按会话/回合失效 | 复用展示，新增独立训练审批域；不持久化现有 BatchGrant 充当长期授权 |
| src/lynx_agent/session/tool_results.py | 工具快照按会话与权限恢复 | 训练结果快照只存ID/摘要，读取日志仍重新检查项目权限 |
| src/lynx_agent/memory/ | MCP/Engram 与来源策略 | 记录成功且有证据的经验，可回查 experiment_id/report_id |

必须先补测试再修改这些入口。只给 TrainingTool.read_only=False 不足以接入：当前 ToolExecutor 的写白名单、工具超时、写操作状态未知转换及重复调用保护都需专项测试。审批等待走受控分支，审批后 RPC 仍有短超时；模型不能通过参数声明自己已批准。

## 3. 包结构与依赖

以下为拟建布局，不在生成本文时创建空源码目录：

```text
src/lynx_agent/training/
  config.py models.py client.py tools.py approval.py presentation.py
services/training_worker/
  pyproject.toml uv.lock
  src/lynx_training_worker/
    cli.py service.py protocol.py config.py
    policy.py discovery.py planner.py workspace.py git.py
    registry.py migrations/ dispatcher.py reconciler.py supervisor.py
    logs.py metrics.py artifacts.py checkpoints.py reports.py
    adapters/base.py adapters/generic.py adapters/<detected_framework>.py
schemas/training/v1/
  request.schema.json response.schema.json plan.schema.json checkpoint.schema.json
tests/training/                 客户端、权限、Schema和集成替身
services/training_worker/tests/  SQLite、调度、状态机、框架替身
tests/fixtures/training/        假项目、日志、检查点清单
tests/manual/                  显式开启的WSL/GPU验收入口
```

Windows 端复用 Pydantic/asyncio；worker 独立 Python 环境，优先使用标准库 sqlite3、asyncio、hashlib、subprocess 和 Pydantic，不依赖 Torch。训练依赖保留在项目环境，由配置指向绝对解释器。服务环境、训练环境和 Windows 环境均锁定版本；新增依赖需审核，不能自动升级现有训练环境。

协议 JSON Schema 是跨包契约源，两侧模型必须通过相同契约测试，不复制后各自漂移。worker 配置与状态留在 WSL，不把 Linux 训练根强行塞入现有 Windows get_home_path()。当前 RAG 独立环境、Engram 和普通会话格式不变。

## 4. 目录与配置契约

管理根默认 `/home/jiahaoxue/lynx-training/`，沿用需求布局。再细分：

- registry/experiments.sqlite3：数据库，目录0700、文件0600；仅管理服务写。
- projects/<id>/capabilities.json：受控发现结果与证据摘要，不含秘密。
- runs/<job_id>/attempts/<attempt_id>/：每次执行的 plan.json、状态回执、脱敏分段日志、指标、控制文件和 checkpoint 索引。
- workspaces/<project_id>/<code_version_id>/：隔离工作副本与验证后的运行快照；输出必须指向 runs，不写进源码快照。
- backups/：SQLite 一致性备份及 schema/版本/校验清单。
- secrets/：控制通道凭据的受保护引用，普通文件与日志工具不可访问；不保存 DeepSeek 密钥。

以下 TOML 是未来配置示例，不得直接写入现有 config.toml：

```toml
[training]
enabled = false
distribution = "Ubuntu-24.04"
root = "/home/jiahaoxue/lynx-training"
rpc_timeout_seconds = 20
max_gpu_jobs = 1
runtime_limit_seconds = "unlimited"
poll_seconds = 10
idle_poll_seconds = 60
max_log_chars = 16000
max_repair_attempts = 3
```

外部 TOML 用字符串 unlimited 表示无固定时限，解析后变为内部 null；不能用0或负数隐式关闭超时。RPC 超时不控制训练时长。磁盘阈值、checkpoint预留空间和日志容量属于管理配置，不是模型可改参数；接入时根据本机余量形成批准计划，缺少阈值不直接开启正式任务。

## 5. 项目接入、需求与代码流程

### 5.1 发现流程

用户提供源码位置 → 确定发行版与规范化根路径 → 静态读取依赖、锁文件、README、AST和配置 → 生成能力候选 → 安全环境探针 → 提交接入计划。

静态阶段不 import 项目、不 source 脚本、不执行 --help，也不执行仓库安装钩子。资源探针只调用已知系统程序，限制输出，不读取全部环境变量；项目解释器路径先作为候选，运行版本探针也需检查实际程序、环境激活机制和潜在启动钩子。发现冲突保留证据和不确定性，不为了自动化猜一个框架。

capabilities 版本化记录：adapter_id/version、发现证据、framework版本、Python解释器、依赖清单摘要、entrypoint/参数Schema、测试套件、指标协议、暂停安全边界、恢复级别、checkpoint验证器、未知项。锁文件中的版本与实际环境版本分别记录。探针不是训练验收。

Windows 源码使用实际挂载解析，不仅替换盘符。Linux realpath 校验允许根、链接、挂载与文件类型；既有特殊路径策略移植到 Linux，遇到未知类型拒绝。只注册显式项目/数据/模型/输出根，不能因为 WSL 可访问 /mnt/c 就拥有整个C盘权限。

### 5.2 代码与验证

需求生成 requirement revision 与受保护验收清单；Agent 在授权范围实现，可有界迭代3次。超过范围、修改验收标准、安装依赖或扩大预算需新批准。测试命令是 argv 数组，测试代码自身也可能有副作用；运行在已批准的临时目录和环境，不默认可信。

有 Git 的源码建立受管本地副本/独立分支工作目录，不切换用户原目录；无 Git 的项目先提出建立版本基线的计划，经批准再做。新建开发分支默认 codex/ 前缀，不改用户原分支；离线本地操作不隐式 fetch/push。开发仓库可创建提交，运行快照不允许模型继续修改。

提交前检查全部待提交文件、已知密钥、忽略规则、子模块/LFS/外部代码依赖和未跟踪文件。配置、数据、模型通过有版本引用接入；不得盲目提交 .env 或把整个大数据集当代码。测试结果绑定 tree/config/environment/suite 摘要；之后任一输入变化使验证失效。绝对依赖原工作区的 editable 安装/PYTHONPATH 必须发现并处理，不能仅改 cwd 就宣称代码隔离。

Git commit 记录通过的验证；运行计划绑定该 commit/tree及外部依赖清单。工作目录隔离不是OS沙箱，不承诺阻止同账户恶意脚本。回滚优先新建 revert commit 或选择历史版本的独立工作目录，保留用户未提交内容、历史台账和检查点。回滚后重新测试；schema不兼容则阻止写启动，不自动覆盖数据库。

## 6. 工具/API设计

模型侧采用稳定工具组，内部操作不全部暴露。所有工具参数 extra=forbid、严格类型校验、UUID与路径约束，拒绝重复JSON键、NaN/Inf、过深嵌套和超大参数。只读与写入操作不得混用 read_only 标记。

| 模型侧工具 | 关键参数 | 内部能力 | 写/批准 |
| --- | --- | --- | --- |
| training_inspect | source 或 project_id，mode=environment/project | inspect_environment、inspect_training_project | 只读；注册根须用户批准 |
| training_plan | project_id, requirement_id, method, baseline_id | plan_experiment、validate_experiment | 创建草稿，不代表批准 |
| training_prepare | project_id, plan_id, action=workspace/tests/git | prepare_workspace、run_project_tests、git_versions | 按作用域批准，不接受任意argv |
| training_submit | project_id, plan_id, request_id | submit_job | 必须宿主审批；不接受模型填写 approval_id |
| training_status | project_id, job_id 或有限过滤 | get_jobs、query_experiments | 只读，最大100条 |
| training_logs | project_id, job_id, attempt_id, cursor, limit | read_job_logs | 只读，limit≤16000 |
| training_control | project_id, job_id, action=pause/resume/cancel | request_pause、resume_job、cancel_job | 每操作有明确批准，幂等 |
| training_compare | project_id, experiment_ids, protocol_id | compare_experiments | 只读；持久化报告另批准 |
| training_records | project_id, action=import/export, reference_id | import_experiment、export_report | 输出/导入范围批准，不接收任意SQL |

inspect_checkpoint 为控制/恢复前的只读内部操作；backup_registry 为运维入口，不暴露为模型任意恢复数据库的工具。框架参数由已注册adapter的Schema提供，训练提交不能传任意Shell。批准后的评测是独立eval job，或预先批准DAG中确定的子节点；模型不能在训练结束后临时换评测命令。

## 7. 控制通道与审批协议

### 7.1 传输

Windows 以 argv 调用已批准的 wsl.exe、固定发行版、worker环境绝对解释器和 client 子命令；JSON 请求走stdin，不拼接 bash -c。client 连接同用户运行目录下0600 Unix socket；服务校验对端UID与项目范围。stdout 仅一个有大小限制的JSON响应，stderr是脱敏诊断；WSL告警需与业务JSON隔离处理，不能当成成功响应。

API版本为整数1；不兼容版本拒绝。客户端支持取消短请求，但取消不能被解释为取消训练。超过20秒返回控制操作结果未知，使用同一个 request_id 查询，不创建新任务；后台持续观察不调用模型。

### 7.2 不可变计划与真实用户批准

Plan包含需求版本、代码版本、环境、adapter版本、argv、cwd、输入版本、输出范围、指标协议、资源与时限、预批准评测依赖。按规定的规范JSON（UTF-8、键排序、无浮点非有限值、无重复键）计算 SHA-256；实际命令中秘密只用credential_ref，不放入计划明文。

现有 ApprovalManager 只负责展示和真实用户回调。新增 TrainingApprovalBroker 不接受模型、网页、日志或训练工具发来的“批准”动作；独立生成带 digest、允许阶段、有效期、nonce与撤销代数的签名凭据。配对密钥在明确安装流程中创建，Windows与WSL各自保护，既不放argv也不返回模型；使用标准HMAC-SHA256并做常数时间校验，不能把现有 BatchPlan.signature() 误认为授权签名。

worker事务内校验签名、范围、有效期、nonce、项目注册代数和当前策略，再绑定具体job。凭据只可消费到该任务/预批准DAG；同request重放返回原结果，不重复消费。摘要变化、项目撤权、凭据过期或环境指纹不符均拒绝。服务端时钟参与有效期判断，时钟偏差报告而非忽略期限。

审批结果可跨会话绑定已提交任务，但不是通用跨回合写授权。普通 /approval revoke 仍只作用于原审批域；新增 /training revoke 明确撤销未执行节点和后续控制授权。对运行中任务撤权不自动强杀，应显示仍运行并由用户另行暂停/停止。审计需同时记录宿主决策与worker接受结果。

这是单用户应用层边界，不提供针对同账户恶意程序的强隔离，也不是密码学意义上证明操作者一定为人。获批任意Shell程序仍可能突破应用保护；后续沙箱属于独立项目，不能在此宣称已具备。

### 7.3 请求与响应示例

下例为内部传输，approval_id/signature由宿主注入，模型工具Schema不提供这些字段；占位标识仅用于说明，不是可运行请求。

```json
{
  "api_version": 1,
  "request_id": "req-example-001",
  "operation": "submit_job",
  "project_id": "project-example",
  "plan_id": "plan-example",
  "plan_digest": "sha256:example",
  "approval_id": "approval-example",
  "approval_signature": "host-generated-placeholder",
  "arguments": {"kind": "train"}
}
```

```json
{
  "api_version": 1,
  "request_id": "req-example-001",
  "status": "accepted",
  "execution_state": "queued",
  "data": {"job_id": "job-example", "attempt_id": "attempt-example"},
  "error": null,
  "observed_at": "2026-10-27T12:00:00Z"
}
```

错误统一含 code、retryable、execution_state、message和evidence_ref。APPROVAL_REQUIRED/PLAN_CHANGED/PATH_DENIED/CAPABILITY_UNSUPPORTED 是未执行；RPC_UNKNOWN 是结果未知；RESOURCE_BUSY 是排队或拒绝，不等于训练失败；CHECKPOINT_INCOMPLETE/RESUME_INCOMPATIBLE 不自动改成新训练。RPC错误消息不含原始命令秘密或完整堆栈。

## 8. SQLite v1数据设计

单服务内一个写入队列，其他线程不得直接写连接。初始化开启 foreign_keys，WSL本地盘使用WAL与FULL同步；事务内只写短数据，不持锁等待训练或模型。busy_timeout初值5秒，超时报告并限制新提交；磁盘满不假报写成功。所有时间UTC，事件序列和state_version用于排序/CAS，不仅依赖墙钟。

以下是可在内存SQLite验证的设计DDL，不是生产迁移脚本；真实迁移须补上schema校验、备份、升级兼容与恢复测试。JSON字段由严格应用Schema验证，跨实体的同项目归属、状态转换和批准逻辑在服务事务中额外校验，不能假定单列FK覆盖所有业务约束。

```sql
PRAGMA foreign_keys = ON;
CREATE TABLE schema_versions (
  version INTEGER PRIMARY KEY CHECK(version > 0), applied_at TEXT NOT NULL
);
CREATE TABLE projects (
  id TEXT PRIMARY KEY, distribution TEXT NOT NULL, source_root TEXT NOT NULL,
  registration_generation INTEGER NOT NULL DEFAULT 1,
  enabled INTEGER NOT NULL CHECK(enabled IN (0,1)), created_at TEXT NOT NULL,
  UNIQUE(distribution, source_root)
);
CREATE TABLE requirements (
  id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id),
  revision INTEGER NOT NULL CHECK(revision > 0), body_json TEXT NOT NULL,
  digest TEXT NOT NULL, created_at TEXT NOT NULL, UNIQUE(project_id, revision)
);
CREATE TABLE code_versions (
  id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id),
  commit_sha TEXT NOT NULL, tree_sha TEXT NOT NULL, workspace_ref TEXT NOT NULL,
  evidence_json TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE test_results (
  id TEXT PRIMARY KEY, code_version_id TEXT NOT NULL REFERENCES code_versions(id),
  input_digest TEXT NOT NULL, suite_digest TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('passed','failed','skipped','unknown')),
  evidence_ref TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE environments (
  id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id),
  interpreter TEXT NOT NULL, fingerprint TEXT NOT NULL,
  snapshot_json TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE plans (
  id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id),
  requirement_id TEXT NOT NULL REFERENCES requirements(id),
  code_version_id TEXT NOT NULL REFERENCES code_versions(id),
  environment_id TEXT NOT NULL REFERENCES environments(id),
  digest TEXT NOT NULL, payload_json TEXT NOT NULL, created_at TEXT NOT NULL,
  UNIQUE(project_id, digest)
);
CREATE TABLE approvals (
  id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id),
  plan_id TEXT NOT NULL REFERENCES plans(id), nonce TEXT NOT NULL UNIQUE,
  scope_json TEXT NOT NULL, receipt_digest TEXT NOT NULL, expires_at TEXT NOT NULL,
  state TEXT NOT NULL CHECK(state IN ('granted','consumed','revoked','expired')),
  created_at TEXT NOT NULL
);
CREATE TABLE experiment_groups (
  id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id),
  name TEXT NOT NULL, baseline_experiment_id TEXT REFERENCES experiments(id),
  comparison_spec_json TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE experiments (
  id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id),
  group_id TEXT REFERENCES experiment_groups(id), plan_id TEXT REFERENCES plans(id),
  method TEXT NOT NULL, provenance_json TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE jobs (
  id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id),
  experiment_id TEXT NOT NULL REFERENCES experiments(id),
  plan_id TEXT NOT NULL REFERENCES plans(id), parent_job_id TEXT REFERENCES jobs(id),
  kind TEXT NOT NULL CHECK(kind IN ('train','eval','probe')),
  state TEXT NOT NULL CHECK(state IN (
    'awaiting_approval','queued','starting','running','pause_requested','checkpointing',
    'paused','resume_requested','cancelling','completed','failed','cancelled','unknown')),
  state_version INTEGER NOT NULL DEFAULT 0, observed_at TEXT, created_at TEXT NOT NULL
);
CREATE TABLE attempts (
  id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id),
  generation INTEGER NOT NULL CHECK(generation > 0),
  unit_name TEXT NOT NULL UNIQUE, boot_id TEXT, invocation_id TEXT,
  pid INTEGER, process_start TEXT, source_checkpoint_id TEXT REFERENCES checkpoints(id),
  gpu_slot INTEGER CHECK(gpu_slot IS NULL OR gpu_slot = 0),
  phase TEXT NOT NULL, exit_code INTEGER, started_at TEXT, ended_at TEXT,
  UNIQUE(job_id, generation)
);
CREATE UNIQUE INDEX one_gpu_owner ON attempts(gpu_slot) WHERE gpu_slot IS NOT NULL;
CREATE TABLE requests (
  id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id),
  operation TEXT NOT NULL, payload_digest TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('pending','accepted','rejected','unknown')),
  response_json TEXT, created_at TEXT NOT NULL
);
CREATE TABLE dispatch_outbox (
  id TEXT PRIMARY KEY, request_id TEXT NOT NULL REFERENCES requests(id),
  attempt_id TEXT NOT NULL UNIQUE REFERENCES attempts(id),
  state TEXT NOT NULL CHECK(state IN ('pending','dispatching','launched','unknown','done')),
  updated_at TEXT NOT NULL
);
CREATE TABLE job_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL REFERENCES jobs(id),
  attempt_id TEXT REFERENCES attempts(id), sequence INTEGER NOT NULL,
  event_type TEXT NOT NULL, source TEXT NOT NULL, payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL, UNIQUE(job_id, sequence)
);
CREATE TABLE metrics (
  id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL REFERENCES experiments(id),
  attempt_id TEXT REFERENCES attempts(id), protocol_id TEXT NOT NULL,
  name TEXT NOT NULL, split TEXT NOT NULL, step INTEGER NOT NULL CHECK(step >= 0),
  value REAL NOT NULL CHECK(value BETWEEN -1.7976931348623157e308 AND 1.7976931348623157e308),
  unit TEXT NOT NULL, evidence_ref TEXT NOT NULL, created_at TEXT NOT NULL,
  UNIQUE(experiment_id, attempt_id, protocol_id, name, split, step, evidence_ref)
);
CREATE TABLE artifacts (
  id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id),
  attempt_id TEXT NOT NULL REFERENCES attempts(id), kind TEXT NOT NULL,
  locator TEXT NOT NULL, digest TEXT, size_bytes INTEGER CHECK(size_bytes >= 0),
  state TEXT NOT NULL CHECK(state IN ('writing','published','invalid','missing')),
  created_at TEXT NOT NULL
);
CREATE TABLE checkpoints (
  id TEXT PRIMARY KEY, artifact_id TEXT NOT NULL UNIQUE REFERENCES artifacts(id),
  manifest_json TEXT NOT NULL,
  resume_level TEXT NOT NULL CHECK(resume_level IN ('full','weights_only','unsupported','unknown')),
  verified INTEGER NOT NULL CHECK(verified IN (0,1)), created_at TEXT NOT NULL
);
CREATE TABLE reports (
  id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id),
  group_id TEXT REFERENCES experiment_groups(id), evidence_json TEXT NOT NULL,
  locator TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE INDEX jobs_by_project_state ON jobs(project_id, state, created_at);
CREATE INDEX events_by_job ON job_events(job_id, id);
CREATE INDEX metrics_by_experiment ON metrics(experiment_id, protocol_id, name, step);
```

approvals只存回执摘要与scope，不存配对密钥。attempt.phase的合法转换由状态机验证；控制类命令也有requests记录。导入历史记录允许 experiments.plan_id 为NULL并在provenance标明unknown，不能补造已批准计划；只有新提交任务才要求完整plans。指标attempt_id为NULL的历史导入去重另使用导入内容摘要，不能依赖SQLite对NULL的唯一性行为。

## 9. 派发、状态机与恢复

### 9.1 唯一提交

1. 校验项目、计划、测试、凭据和资源；同request_id不同payload_digest返回冲突。
2. BEGIN IMMEDIATE内消费批准并创建request、job、attempt、outbox；全部成功后提交，未提交前不启动程序。
3. Dispatcher认领outbox并预约唯一gpu_slot；固定unit名由attempt_id派生，带完整的受管标签。
4. Supervisor请求systemd启动固定wrapper，wrapper校验已提交manifest后按argv执行训练。
5. 启动回执或事件更新attempt与job，outbox转launched；短请求超时仍可用原request_id查回结果。

此设计追求可对账的单次派发，不宣称SQLite与systemd之间存在分布式原子事务或绝对exactly-once。systemd回执丢失时先查询相同unit/InvocationID、wrapper启动/退出证据；不能自动改unit名再执行。相同名称的外部unit必须拒绝接管。已有进程是否真正运行和是否产生副作用不明时标unknown，保留资源槽并阻止重复启动，交由协调或人工解决。

### 9.2 核心转换

| 当前状态 | 条件/事件 | 下一状态 | 副作用 |
| --- | --- | --- | --- |
| awaiting_approval | 用户批准且计划匹配 | queued | 写事务/outbox |
| queued | 资源与前置条件满足 | starting | 预约槽并派发 |
| starting | 受管启动证据核实 | running | 记录身份 |
| running | 用户请求暂停、adapter支持 | pause_requested | 写控制请求 |
| pause_requested | 训练回调接受安全边界请求 | checkpointing | 保存临时checkpoint |
| checkpointing | checkpoint验证成功且进程有序退出 | paused | 发布清单并释放GPU槽 |
| paused | 新批准与兼容性通过 | resume_requested → queued | 同逻辑job创建新attempt |
| running | 退出码0且运行产物契约通过 | completed | 登记训练完成，独立判定评测 |
| 活动态 | 已核实异常退出 | failed | 保存退出证据、释放槽 |
| queued/活动态 | 用户批准停止并核实退出 | cancelled | 不声称保存了checkpoint |
| 任意未终态 | 进程/回执无法核实 | unknown | 不释放不确定资源、不重提 |

completed/failed/cancelled为job终态，不受迟到心跳覆盖；paused为可恢复停驻态，旧attempt已结束。resume不能复用旧attempt_id。所有事件携带attempt_id/generation/state_version，worker忽略旧generation并记录异常。state_version CAS冲突重新读状态，不覆盖新事件。

GPU槽只有确认受管进程退出且相关worker已清理后释放，不能只看心跳过期。预先声明的eval子任务在训练完成且指定checkpoint发布后进入队列；使用批准的输出选择器和评测协议，不接受模型临时返回一个任意checkpoint路径。失败父任务阻断依赖，不能把旧检查点冒充本轮产物。

### 9.3 systemd托管与存活

worker和任务使用不同用户unit；任务wrapper保留规范化argv、cwd、允许环境与退出回执，训练不会因client/TUI退出被清理。GPU任务异常不得由systemd自动重启，服务可重启但必须先对账，不能重新执行所有pending数据。进程归属记录distribution、boot_id、unit、InvocationID、PID与启动时间，不只用PID。

用户级服务可用性、WSL生命周期、退出终端和睡眠行为必须在目标机器实测。systemd不是WSL永久保活保证；安装方案须验证独立于TUI的生命周期锚定与健康检查，失败则禁止正式任务。不会自动设置root服务、修改电源策略或启用linger；需要时明确申请授权。WSL关闭、宿主机重启/断电可能中断训练，只承诺依赖有效checkpoint的恢复能力，不承诺不停机。

## 10. 暂停、检查点与续训适配

### 10.1 Adapter契约

FrameworkAdapter提供 discover、validate_config、build_command、capabilities、read_metrics、request_checkpoint_stop、validate_checkpoint、build_resume_command。适配器部署在worker侧，但checkpoint回调执行在训练环境内；回调为经批准的小模块，不依赖完整Agent或SQLite库。

capabilities必须明确训练步/梯度累积/rollout的安全边界、完整恢复所需文件、分布式rank协调与最大正常等待预期。generic adapter可运行已批准入口及观察日志，不支持的pause返回CAPABILITY_UNSUPPORTED；不得偷偷SIGTERM。实际框架未识别前不能编写声称已适配的通用GRPO停止回调。

### 10.2 合作式暂停

worker写入带request_id/attempt_id/nonce的原子控制文件 → 训练回调轮询并确认 → 在框架认可的安全边界请求保存 → 所有必要rank完成 → 在临时目录校验并写manifest → 原子发布完成标志 → orderly exit → wrapper与worker确认暂停。

控制文件是请求，不是保存成功证据。多个rank不能同时发布冲突manifest。框架原生保存无法改目录协议时，adapter在原生保存完成且验证后建立受控发布标识；只读取有完成标识并经验证的产物。不能用“看到checkpoint目录”或日志含saved当充分证据。

manifest包含format_version、job_id、attempt_id、global_step、code/config/environment/adapter摘要、模型和tokenizer引用、dataset/reward/eval协议、文件清单/大小/哈希、optimizer/scheduler/RNG/data进度可用性、resume_level、完成时间。文件哈希可后台完成，但未完成验证前不可发布为full。元数据不保存密钥。

保存失败或等待超时：记录pause_failed，保留上一有效checkpoint和真实进程状态；若训练继续则恢复running观察，用户可继续等待或单独批准停止。不强杀来制造“暂停完成”。退出与checkpoint发布之间崩溃由对账器核实，证据不全为unknown。

### 10.3 恢复规则

full要求框架验证必要状态完整，包括数据进度、优化器、调度器、随机数及框架特有状态；weights_only只允许以明确的新训练计划加载，不能叫续训。检查Manifest再做授权的兼容性探针，不反序列化不可信pickle来做普通只读检查。

恢复使用同job的新attempt，source_checkpoint_id与新generation留痕；配置、代码、数据或环境变化须进行兼容性判断并新批准，不默认绕过。GRPO需具体检查rollout、奖励实现/依赖和采样后端状态；不能以SFT恢复测试代替。验收比较连续跑与暂停续跑的步数、学习率、optimizer状态、样本进度和批准的容差，不默认要求GPU浮点逐位相同。

## 11. 日志、指标、比较与记忆

日志按attempt分段，控制台内容按不可信文本处理并移除危险控制序列。已知密钥在持久化和外发前按跨分块边界脱敏，raw未脱敏流不写普通日志；上游已有日志只能在读取/导入时脱敏，不能承诺消除原文件秘密。游标包含segment、字节偏移与generation；UTF-8边界安全。轮转后旧游标返回CURSOR_EXPIRED及可用区间，不静默跳过。

单次最多16,000字符，分页返回next_cursor/truncated/evidence_ref；模型上下文不等于全日志存储。长内容可经现有工具回收，但恢复日志需按project权限再验；会话快照不授予跨项目数据读取。默认10秒观察、无变化退避到60秒，错误退避；解释只在用户询问或预批准事件触发。

指标优先解析注册的JSONL/标准格式，禁止用模型猜数值。确定性适配转换后保存名称、单位、step、split、protocol_id、原始证据与采样时间；NaN/Inf标异常，不转成0。模型解释存report且注明推断，不修改metrics。

GRPO/SFT对比需固定共同eval协议与测试集版本，列明基模、训练预算、种子、代码、配置差异，分别展示reward/loss。单次跑只报告观察差异，不声称统计显著；若多次独立运行按预先指定规则聚合，不能训练后挑最好值。未知/缺失历史字段显式显示，不能把无法核实的结果补全为受控对照。

SQLite先保存事实和证据，再可选通过现有MemoryService异步提取经验，绑定project/experiment/report/commit；记忆写入用幂等键重试且显示pending/failed，不阻塞任务完成。RAG仍按知识库独立授权；不能因引入training就自动导入日志、论文或数据集。DeepSeek可接收任务必要资料，但凭据保护和有界上下文持续有效。

## 12. 错误处理、备份与运维

| 故障 | 行为 |
| --- | --- |
| DeepSeek不可用 | 任务与事实采集继续；解释延后，不无限重试API |
| client超时 | 查询原request_id，不能重新启动 |
| worker重启 | 加写者锁、校验schema、先对账再开放写提交 |
| SQLite锁/损坏/磁盘满 | 限制新写/提交，保存可获取现场，不伪报成功；现有任务可能继续 |
| systemd回执丢失 | 查询受管unit和启动证据，未知保留槽 |
| checkpoint不足或不兼容 | 阻止full resume，明确恢复级别 |
| Engram不可用 | 台账正常，记忆同步待处理 |
| 文件或项目撤权 | 阻止新读取/提交与快照恢复，明确运行任务仍存在 |

使用SQLite在线备份接口生成一致性副本，备份后在隔离副本运行完整性与FK检查，记录schema、时间与校验值；不能仅复制WAL模式的主db文件。初始建议每日一份及每次迁移/显式恢复前备份，不自动删除历史，容量不足提示用户处理。恢复数据库前停止接收写请求并与受管任务对账，未经确认不覆盖当前库，恢复不能让旧outbox自动重放。

迁移按schema版本显式执行，备份成功后BEGIN事务；失败回滚并停止发布新版本。应用声明可读/可写schema范围，不兼容版本只读或拒绝启动。代码回滚与数据恢复分为两项批准，恢复备份后仍需对账现实中的运行进程与产物。

默认只记录计数、ID、状态与耗时，提供health、outbox积压、未知任务数、最后心跳、GPU槽、磁盘余量、台账写失败、checkpoint失败和记忆同步积压。仪表不输出请求秘密、完整代码或隐藏推理。保留率/日志大小可配置，首版不自动删检查点。

安装拆分为预检 → 用户确认文件与服务变更 → 建立worker环境 → 部署用户服务/控制凭据 → 协议握手 → 临时假任务验收 → 显式GPU验收。卸载先确认运行任务归属和停止策略，仅移除明确服务与软件，不删除台账、日志、原仓库和checkpoint。

## 13. UI与配置集成

拟新增 /training status|projects|jobs|show|logs|pause|resume|cancel|compare|revoke 命令；均待实现，不在本次加入CLI。状态面板显示发行版、项目、job/attempt、阶段、观测时间、GPU槽和是否有可恢复checkpoint；不把模型“正在思考”当训练运行状态。

用户取消当前模型请求只取消控制等待，不默认停止训练。任务操作命令包含准确job_id，审批卡展示代码/配置/输入/输出、资源、无限时长提示和后续评测范围；同名标题不代替ID。headless只允许查询或使用预先批准且未过期的准确任务计划，不能通过参数自动批准。

现有 /permissions read、/approval paused 下不允许新提交、代码写入或恢复；只读观察须有项目授权。暂停/停止属于受管副作用，在常规写入暂停时可由独立人工紧急控制入口明确批准执行，不能使模型自动获得例外。普通 /shell off 不应停止已存在训练，但也不允许借训练工具变相执行任意Shell。

## 14. 需求追踪、实现顺序与验收

| 需求 | 主要模块 | 核心验证 |
| --- | --- | --- |
| F01 | discovery/registry | 源码接入、跨Session查询、未知历史字段 |
| F02 | policy/planner/adapter | 配置、资源、路径和版本门禁 |
| F03 | approval/dispatcher/outbox | 幂等、批准变更、崩溃注入 |
| F04 | supervisor/logs/events | 真实状态、游标、脱敏、迟到事件 |
| F05 | worker/reconciler | TUI关闭与worker重启后对账 |
| F06 | metrics/artifacts/checkpoints | 指标证据、完整发布、非有限数 |
| F07 | comparison/reports | 共同eval、历史不可比、不挑最好结果 |
| F08 | controls/adapters | checkpoint退出、兼容续训、目标任务归属 |
| F09 | diagnostics/repair | 有界修复、不可降标准、无GPU盲重试 |
| F10 | reports/registry | 导出权限、历史证据、凭据保护 |
| F11 | requirements/workspace/tests | 需求到差异、验证、commit链 |
| F12 | git/migrations | 非破坏回滚、脏文件保护、schema不兼容 |

M1：协议/Pydantic模型/发现/路径/权限/内存SQLite迁移与台账；M2：需求与隔离代码、测试及Git闭环；M3：worker、审批、systemd替身、outbox与状态对账；M4：实际框架的SFT/GRPO回调和checkpoint协议；M5：评测/对比/报告/记忆引用/TUI；M6：真实WSL与GPU门禁。每步测试通过再commit，不先建大量空模块。

自动测试需覆盖需求A01—A12，外加协议版本、JSON重复键、HMAC伪造/过期/撤销、幂等payload冲突、GPU槽竞争、各派发崩溃点、PID复用、事件乱序、暂停竞态、UTF-8与分块秘密、日志轮转、SQLite FK/迁移/备份、回滚保留用户修改。使用临时目录和fake supervisor/adapter，不操作用户现有训练。

WSL测试显式开关，检查跨系统参数/路径、用户服务、关闭TUI后的存活、终止与重启对账；GPU测试另设显式开关，在微型数据与模型上分别验证实际SFT和GRPO的训练、共同eval、暂停及续训。项目框架未知前不能确定真实测试配方，获得源码后由discovery生成并由用户批准。未执行的验收必须skip并报告原因，不能以替身测试代替。

文档中的DDL/JSON样例应有离线契约检查；SQL可执行和文档链接通过仅代表设计自洽，不证明系统已具备这些功能。主回归继续执行AGENTS要求的pytest、Ruff及Git差异检查；未来worker的独立测试还需加入交付门禁。

## 15. 实现前仍需核验的事项

唯一必需用户输入是训练源码位置与任务目标。框架/入口/环境/历史指标/恢复支持由Agent读项目查明；有歧义的业务指标、数据划分和权限扩大再请用户决定。不能自动迁移正在运行的实验、改已有训练脚本或创建服务来“验证设计”。

以下参考为实现核验入口。本轮尝试联网重取失败，故未声称逐项验证最新API；实际实现须锁定对应版本并验证系统行为：

- [WSL systemd与生命周期说明](https://learn.microsoft.com/en-us/windows/wsl/systemd)。
- [systemd-run用户任务选项](https://www.freedesktop.org/software/systemd/man/latest/systemd-run.html)。
- [SQLite WAL约束](https://www.sqlite.org/wal.html)与[在线备份](https://www.sqlite.org/backup.html)。
- [Transformers callback接口](https://huggingface.co/docs/transformers/main_classes/callback)与[checkpoint恢复](https://huggingface.co/docs/transformers/trainer_recipes)。

当前交付只新增技术文档与文档契约测试，不创建worker包、训练目录、数据库或unit，不提交凭据；当前已交付功能仍以[产品方案](product-plan.md)为准。
