# 分级审批测试方案

本文给出可复现的测试方案：先用自动化测试覆盖权限、授权生命周期、路径和网络边界，再用 TUI 手工验收确认卡片、按键、进度和停止按钮。所有测试只使用临时目录、模拟网络和可控子进程，**不要使用真实私人资料**。

> 测试对象是应用层权限与审批机制，不是操作系统沙箱。测试通过不代表能阻止同账户恶意进程、获批程序的任意文件访问或网络外传。

## 1. 环境准备

在项目根目录 `E:\github\nanobot` 打开 PowerShell：

```powershell
$env:UV_CACHE_DIR='E:\github\nanobot\.uv-cache'
uv sync --locked
```

建议先确认当前配置和测试基线：

```powershell
uv run --locked pytest -q
uv run --locked ruff check
```

当前预期基线：**768 passed, 11 skipped**；`ruff` 输出 **All checks passed**。跳过项来自平台/外部服务条件，不是审批功能失败。

## 2. 自动化测试关卡

### 关卡 A：全量回归（必跑）

```powershell
$env:UV_CACHE_DIR='E:\github\nanobot\.uv-cache'
uv run --locked pytest -q
uv run --locked ruff check
```

通过标准：

- `pytest` 显示 `768 passed, 11 skipped`（新增测试后；若你继续加测试，数字会增长，但 skipped 不应无故增加）。
- 没有任何 failed/error。
- `ruff` 没有输出错误。

### 关卡 B：审批专项（快速定位）

```powershell
uv run --locked pytest tests/approval -q
```

预期：**38 passed**。

按子领域运行：

```powershell
uv run --locked pytest tests/approval/test_policy.py -q
uv run --locked pytest tests/approval/test_batch_tools.py -q
uv run --locked pytest tests/approval/test_download_extract.py -q
uv run --locked pytest tests/approval/test_tui_approval.py -q
uv run --locked pytest tests/approval/test_agent_batch.py -q
uv run --locked pytest tests/approval/test_execution_binding.py -q
uv run --locked pytest tests/approval/test_config.py -q
```

### 关卡 C：原有保护回归

```powershell
uv run --locked pytest tests/filesystem tests/execution -q
uv run --locked pytest tests/cli/test_permissions.py tests/cli/test_tui.py tests/cli/test_headless.py -q
uv run --locked pytest tests/providers/test_output_guard.py tests/web/test_security.py -q
```

这些测试覆盖文件边界、Shell 审批、进程树清理、密钥脱敏、权限点击切换、草稿保留和网络地址限制。

## 3. 自动化覆盖矩阵

| 需求 | 自动化测试 |
| --- | --- |
| 默认“工作区修改 + 常用操作自动” | `test_policy.py::test_defaults_are_auto_and_labels_are_explicit` |
| 两套开关独立，完全权限不取消审批 | `test_policy.py::test_runner_defaults_and_independent_switches`、`test_tui_approval.py::test_approval_selector_is_independent_from_file_permissions` |
| 自动只读不反复询问 | `test_policy.py::test_auto_reads_do_not_ask_and_paused_allows_analysis` |
| 每次询问档位 | `test_policy.py::test_ask_mode_asks_for_reads_and_new_files` |
| 暂停执行可分析但不可修改/提交 | `test_policy.py::test_paused_allows_preview_but_refuses_commit_and_new_files`、`test_modes_validate_and_paused_blocks_execution_service` |
| 无人审批入口不执行 | `test_policy.py::test_no_presenter_denies_ask_and_batch` |
| 批量任务一次授权后连续执行 | `test_batch_tools.py::test_batch_replace_one_card_then_reuses_grant`、`test_agent_batch.py::test_agent_batch_calls_use_one_authorization_and_complete` |
| 授权随回合/会话/权限失效 | `test_batch_tools.py::test_batch_grant_expires_on_turn_session_and_permission_change` |
| 新目录/新操作重新授权 | `test_batch_tools.py::test_new_directory_or_operation_requires_new_card` |
| 超限阻止且不产生修改 | `test_batch_tools.py::test_batch_limits_block_before_any_change` |
| 模型不能伪造批准字段 | `test_batch_tools.py::test_model_cannot_forge_approval_fields` |
| 拒绝后不能换工具重试 | `test_batch_tools.py::test_denied_batch_cannot_be_replayed_with_another_tool` |
| 覆盖前备份，外部修改跳过 | `test_batch_tools.py::test_external_change_is_skipped_without_overwrite` |
| 停止后如实报告部分修改 | `test_batch_tools.py::test_stop_reports_partial_changes_without_rollback` |
| 外部目录单独授权，不要求完全权限 | `test_batch_tools.py::test_batch_can_approve_specific_external_directory_without_full_permission` |
| 只读/暂停时批量工具不可用 | `test_batch_tools.py::test_batch_tools_unavailable_in_read_only_or_paused` |
| 批量工具拒绝敏感凭据 | `test_batch_tools.py::test_batch_tools_refuse_sensitive_and_private_paths` |
| 批量卡片长清单分页 | `test_batch_tools.py::test_card_pages_long_file_list`、`test_tui_approval.py::test_batch_card_supports_batch_decision_and_paging` |
| 下载来源授权、重定向复验、不执行 | `test_download_extract.py::test_download_requires_source_authorization_and_does_not_execute`、`test_download_redirects_are_revalidated` |
| 下载大小限制、同名不覆盖 | `test_download_extract.py::test_download_size_limit_and_target_conflict` |
| 下载协议/凭据/私网限制 | `test_download_extract.py::test_download_url_validation_rejects_protocol_credentials_and_private_host` |
| 解压越界、链接、重名、超限 | `test_download_extract.py::test_extract_zip_rejects_traversal_links_and_duplicates`、`test_extract_total_size_limit_blocks_archive` |
| 解压只写授权目录，已存在跳过 | `test_download_extract.py::test_extract_writes_only_inside_destination_and_skips_existing` |
| 归档审批后被修改则作废 | `test_download_extract.py::test_extract_archive_change_after_approval_invalidates` |
| 下载与解压需要分别授权 | `test_download_extract.py::test_download_then_extract_needs_separate_authorization` |
| 命令审批绑定脚本内容 | `test_execution_binding.py::test_script_change_after_approval_invalidates_command` |
| 审计脱敏 | `test_policy.py::test_audit_redacts_known_secrets_and_never_records_content` |
| 系统目录/Agent 规则保护 | `test_policy.py::test_system_and_agent_rule_writes_blocked_even_full` |
| TUI 的 Y/B/N、翻页、[停止] | `test_tui_approval.py::test_batch_card_supports_batch_decision_and_paging` |

## 4. TUI 手工验收

### 4.1 准备隔离演示目录

演示目录放在固定工作区内，避免使用工作区外真实资料：

```powershell
$demo = 'E:\github\nanobot\approval-demo'
Remove-Item -Recurse -Force $demo -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path $demo | Out-Null
"hello alpha" | Set-Content "$demo\a.txt" -Encoding utf8
"hello beta"  | Set-Content "$demo\b.txt" -Encoding utf8
```

启动：

```powershell
$env:UV_CACHE_DIR='E:\github\nanobot\.uv-cache'
uv run --locked lynx-agent
```

### 4.2 默认状态和普通只读

1. 看底栏：应显示 `权限：工作区修改`、`审批：自动`、`Shell：开启（无沙箱）`。
2. 输入：`列出 approval-demo 目录并读取 a.txt 前 5 行`。
3. 预期：只读工具直接执行，不出现审批卡片；回答基于文件内容。
4. 点击 `审批：自动 ▾` 或输入 `/approval`，确认可选“每次询问 / 常用操作自动 / 暂停执行”；点击“暂停执行”，再做只读查询应仍可分析，但修改会被拒绝。测完切回“常用操作自动”。

### 4.3 批量文本替换：一次批准

输入（明确要求使用固定工具，避免模型改用命令）：

```text
请用 batch_replace_text 把 approval-demo 下 a.txt 和 b.txt 中的 hello 全部替换为 hi；
先展示批量审批卡片，不要用 exec_command，也不要自己假定已批准。
```

如果模型没有调用指定工具，直接纠正它：“请改用 batch_replace_text，不要用 exec_command”；模型行为属于额外变量，授权边界本身以自动化测试为准。

预期：

1. 出现批量卡片，至少包含：任务说明、读取目录、写入目录、操作类型、文件清单、预计数量/总大小/执行上限、覆盖“是”、移动/删除“否”、备份位置、授权有效范围。
2. 文件清单超过 10 个时可 `PgUp/PgDn` 翻页。
3. 按 `B`（允许本批次）后执行；`N`（拒绝）应不修改文件；`Y`（允许一次）只执行本次。
4. 按 `N` 后，再要求“换成 write_file 写入同一文件”应被拒绝（本回合拒绝回溯）。
5. 按 `B` 后检查文件：

```powershell
Get-Content "$demo\a.txt"
Get-Content "$demo\b.txt"
Get-ChildItem -Recurse 'E:\github\lynx_agent_data\approval\backups' -ErrorAction SilentlyContinue
```

预期 `a.txt`/`b.txt` 变成 `hi ...`，备份目录中存在原始内容。备份路径取决于会话存储位置；若使用自定义存储，查看其父目录下的 `approval\backups`。

6. **同一用户消息内**再连续调用同一批量工具（例如一次提示里要求先替换 a.txt 再替换 b.txt）应复用授权，不再重复弹卡；如果模型没有在一次回答里连续调用，用 `tests/approval/test_agent_batch.py` 验证该路径。**新的用户消息是新任务回合，会重新询问**，这是预期行为。
7. 重启程序后再发起同样批量操作，应重新出现卡片（授权不持久化）。

### 4.4 权限和审批失效

1. 批量任务执行前/后切换 `/permissions read`，再要求写入：应被拒绝。
2. 切回 `/permissions workspace`，再切换 `/approval paused`：读取可继续，修改/下载/命令不可用；尝试提交文件覆盖应被拒绝。
3. 切回 `/approval auto`，用 `/new` 切换会话，再发起批量操作：应重新弹卡。
4. 停止按钮：准备 60 个文件后发起一次批量替换，在进度显示时点击底栏 `[停止]` 或按 `Ctrl+X`；预期返回 `stopped`、成功/失败/跳过数量和已修改文件清单，不声称自动回滚。未修改的文件保持不变。

```powershell
$demo2 = 'E:\github\nanobot\approval-demo2'
Remove-Item -Recurse -Force $demo2 -ErrorAction SilentlyContinue
New-Item -ItemType Directory $demo2 | Out-Null
1..60 | ForEach-Object { "hello $_" | Set-Content "$demo2\$_.txt" -Encoding utf8 }
```

### 4.5 下载与解压

1. 下载（公网可用时）：

```powershell
New-Item -ItemType Directory "$demo\downloads" -Force | Out-Null
```

```text
请用 download_file 下载 https://example.com/robots.txt 到 approval-demo/downloads/，
文件名 robots.txt；展示来源卡片，不要执行下载的文件。
```

预期：卡片显示来源主机和端口、大小上限、上传“否”；批准后文件保存，结果标记 `executed: false`，不会自动运行。

2. 解压：先手工创建 ZIP 和输出目录（不要用 Agent 创建）：

```powershell
New-Item -ItemType Directory "$demo\extracted" -Force | Out-Null
$zip = "$demo\bundle.zip"
Remove-Item -Force $zip -ErrorAction SilentlyContinue
Compress-Archive -Path "$demo\a.txt", "$demo\b.txt" -DestinationPath $zip
```

然后要求：

```text
请用 extract_archive 把 approval-demo/bundle.zip 解压到 approval-demo/extracted，
先展示卡片；不要覆盖已有文件。
```

预期：卡片显示归档格式、成员数、总大小、覆盖“否”；批准后只写入 `extracted` 目录。

3. 越界 ZIP（安全负例）：

```powershell
$bad = "$demo\evil.zip"
python -c "import zipfile; z=zipfile.ZipFile(r'$bad','w'); z.writestr('../evil.txt','bad'); z.close()"
```

要求 Agent 解压 `evil.zip`：应拒绝，且 `E:\github\nanobot\evil.txt` 不应出现。测试后删除 `evil.zip`。

### 4.6 敏感文件与保护目录

```powershell
"FAKE_SECRET=not-a-real-secret" | Set-Content "$demo\.env.demo" -Encoding utf8
```

1. 要求读取 `approval-demo/.env.demo`：应只生成敏感路径授权，不直接返回正文；`/approvals` 查看后 `/approve`，再重新请求才读取。测试后删除该文件。
2. 要求读取 `C:\Windows\win.ini`：即使 `/permissions full` 也应被拒绝（系统目录保护）。
3. 要求修改 `config.toml`、`src\lynx_agent\approval` 或 `AGENTS.md`：应被拒绝（Agent 规则保护）。
4. 目录联接/硬链接负例（PowerShell 管理员/开发者模式可能需要）：

```powershell
cmd /c mklink /J "$demo\linked" "$demo"
New-Item -ItemType HardLink -Path "$demo\hard.txt" -Target "$demo\a.txt" | Out-Null
```

要求读取 `linked\a.txt` 或 `hard.txt`：应拒绝。清理：

```powershell
cmd /c rmdir "$demo\linked"
Remove-Item "$demo\hard.txt" -Force
```

## 5. 网络边界的手工负例

不需要真实联网也能验证 URL 校验：

1. 要求 `download_file` 下载 `file:///C:/Windows/win.ini`：应拒绝。
2. 要求下载 `http://127.0.0.1:8000/x`：生产校验应拒绝私网/回环地址（只有测试代码使用 `allow_private=True`）。
3. 要求下载带账号密码的 URL：应拒绝。
4. 若需要验证“重定向到未授权主机”，直接运行自动化测试，不要在生产配置中放宽私网限制：

```powershell
uv run --locked pytest tests/approval/test_download_extract.py::test_download_redirects_are_revalidated -q
```

## 6. 手工检查 Shell 和原有保护

1. 启动 TUI，确认 Shell 默认开启；要求执行 `python -c "print(1)"`，应出现命令卡片，按 `Y` 才执行。
2. 拒绝一次命令，再要求“换个写法执行同一命令”：模型不应获得自动批准；每个新命令仍需新卡片。
3. 打开命令卡片时检查：参数数组、工作目录、超时、权限、会话、直接脚本内容 SHA-256；无“允许本批次”选项。
4. 权限选择：点击 `权限：工作区修改 ▾`，切换仅查看/完全权限，确认草稿和光标保留，切换期间文字不穿透、Enter 不发送。
5. 输入一段草稿后切换审批档位，确认草稿保留；审批档位不改变文件权限。
6. `/shell off` 后命令工具不可用，`/shell on confirm` 需再次确认；`/approval paused` 时不能开启 Shell。
7. 代理/密钥/超时/清理：由 `tests/execution` 和 `tests/stress/test_shell_pressure.py` 覆盖；后者需要显式环境变量：

```powershell
$env:LYNX_PRESSURE_TEST='1'
uv run --locked pytest tests/stress/test_shell_pressure.py -q -s
Remove-Item Env:LYNX_PRESSURE_TEST
```

## 7. 通过标准和证据

一次验收至少保存：

- 全量 `pytest` 输出（passed/skipped 数字）。
- `ruff check` 输出。
- TUI 批量卡片截图或终端记录（包含文件清单、覆盖/备份、授权范围）。
- 批量替换前后的文件内容、备份目录列表。
- 拒绝、暂停、权限切换、重启后的行为记录。
- 下载/解压结果和负例错误消息。

通过标准：

- 自动化数字符合基线，无 failed/error。
- 普通只读不反复询问；批量任务批准一次后能在同一任务连续执行。
- 暂停、仅查看、拒绝、会话切换、重启、权限变化都会阻止后续操作。
- 伪造审批字段、越界路径、链接、系统目录、敏感文件、未授权下载来源均被拒绝。
- 下载后不自动执行；解压不写出授权目录。
- Shell 默认开启、点击切换权限、草稿保留、代理变量、密钥脱敏、超时和进程清理不回归。

## 8. 清理

```powershell
Remove-Item -Recurse -Force 'E:\github\nanobot\approval-demo','E:\github\nanobot\approval-demo2' -ErrorAction SilentlyContinue
```

审计和备份默认位于 `E:\github\lynx_agent_data\approval`，不会自动删除；确认不再需要后再手工清理。不要删除真实会话、记忆或用户文件。

## 9. 常见问题

- **数字不是 768/11**：确认使用 `uv run --locked`，且没有额外安装测试插件或修改环境变量；新增测试会让 passed 增加。
- **TUI 按键无反应**：确认焦点在审批卡片，且终端没有拦截 `B`/`PgUp`/`PgDn`；也可用 `/approvals`、`/approve <编号>`、`/reject <编号>` 完成同样操作。
- **批量操作没有弹卡**：先确认`/approval` 不是 `paused`，再确认目标在授权目录和上限内；同一任务回合已授权时会复用。
- **新用户消息又弹卡**：这是设计行为；授权只在当前任务回合有效，新消息是新任务。
- **手工测试改坏了文件**：从 `approval\backups` 按审计记录人工恢复；没有自动回滚。
