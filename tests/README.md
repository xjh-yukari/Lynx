# 测试与人工调试入口

统一存放自动测试、手工检查、固定语料和历史报告。正式入口 `scripts/rag_runtime.py` 不属于测试程序，保留原路径，确保 `lynx` 快捷启动不受影响。

```text
tests/
  agent/ approval/ cli/ config/ execution/ filesystem/ knowledge/ mcp/ memory/ providers/ session/ tools/ web/
  stress/                     显式开启的压力测试
  manual/                     人工运行，不由pytest自动执行
  fixtures/long_context/      六段上下文测试文本
  fixtures/document_rag/      三份人工文档、JSON映射示例、独立验证授权/配置/清单
  plans/                      审批、长期记忆与RAG测试方案
  reports/                    已保留的历史报告，不是当前测试结论
```

以下命令从 `E:\github\nanobot` 执行，除特别说明外使用根虚拟环境。先查看脚本说明，不自动运行所有人工脚本。

| 命令 | 用途与副作用 |
| --- | --- |
| `uv run --locked pytest -q` | 普通离线回归；重型实测/压力默认跳过 |
| `uv run --locked ruff check src tests` | 代码检查，不调用模型 |
| `uv run python tests/manual/demo_compaction.py` | 临时会话、模拟摘要；不证明真实摘要质量 |
| `uv run python tests/manual/demo_tui.py --tools` | 离线TUI与计算器演示，临时目录，不读真实密钥 |
| `uv run python tests/manual/tui_preview.py` | 打印真实布局纯函数输出，不启动交互界面 |
| `uv run python tests/manual/check_training_foundation.py` | 临时M1项目/SQLite冒烟，不执行训练源码、不调用模型/GPU |
| `uv run python tests/manual/check_memory.py` | 连接真实Engram，不调模型；首次可能创建数据库/运行元数据 |
| `uv run python tests/manual/engram_probe.py E:/github/lynx_agent_data/bin/engram/3.4.0/engram.exe` | 临时目录验证真实Engram协议，向临时库写测试记录 |
| `uv run python tests/manual/check_web.py --search "Python official documentation"` | 真联网，搜索服务可能计费，不调用模型 |
| `uv run python tests/manual/check_web.py --fetch https://www.python.org/` | 真联网抓取，不调用模型 |
| `python tests/manual/tui_mockups.py` | 设计图生成器，需可选Pillow；会覆盖参考PNG，平时不要执行 |

## 真实RAG检查（独立环境）

测试数据库须已就绪，且不能与使用同一 `validation-state` 的查询服务并行。下面会构建/发布**隔离验证库**，不操作正式 `deepseek_docs`；不要当作纯只读健康检查。

```powershell
& E:\github\nanobot\services\document_rag\.venv\Scripts\python.exe E:\github\nanobot\tests\manual\validate_rag_real.py
```

验证配置、授权与清单位于 `tests/fixtures/document_rag/`。报告写入 `tests/reports/rag-last-real-report.json`。端口19540、项目 `rag_validation`；不操作课程19530实例。

容器重启后只读检查alias/数据（依赖对应历史报告，不主动load/drop）：

```powershell
& E:\github\nanobot\services\document_rag\.venv\Scripts\python.exe E:\github\nanobot\tests\manual\check_rag_persistence.py
```

完整本地链路使用真实RAG/Engram、脚本模型替身：

```powershell
$env:LYNX_RAG_REAL='1'
uv run --locked pytest tests/knowledge/test_full_local.py -q
Remove-Item Env:LYNX_RAG_REAL
```

## 压力测试

```powershell
$env:LYNX_PRESSURE_TEST='1'
uv run --locked pytest tests/stress -q -s
Remove-Item Env:LYNX_PRESSURE_TEST
```

会产生真实临时子进程、临时日志和较大输出，Shell压力含50轮取消/超时及40秒以上审批等待。只在pytest临时目录进行，不调用真实模型。

## 语料与历史记录

`fixtures/long_context/part-1.txt` 到 `part-6.txt`用于手工发送。`fixtures/document_rag/session.md` 同时被当前正式导入清单明确引用；不能因目录名是tests就随意删除。

移动语料没有自动重建Milvus。旧报告、旧索引中的来源路径是历史快照，不回写伪造；下一次正式建库使用已更新的 `E:\github\lynx_agent_data\rag\import.json`。

审批、长期记忆、文档检索详细方案分别见 `plans/approval-test-plan.md`、`plans/long-term-memory.md`、`plans/document-rag.md`。产品能力与安全边界统一见 [产品方案](../docs/product-plan.md)。

训练助手自动测试：`uv run --locked pytest tests/training -q`；worker标准库源码额外检查：`uv run --locked ruff check services/training_worker`。真实WSL临时冒烟及后续GPU门禁见[接入计划](plans/training-onboarding.md)和[worker说明](../services/training_worker/README.md)，没有默认真实训练测试。

## DeepSeek 直连回归

公开发布准备：`uv run --locked pytest tests/test_publication.py -q`覆盖新用户示例配置、文档链接、忽略规则以及索引/历史敏感信息检查；扫描器检查`uv run --locked ruff check scripts/check_publication.py`。发布前另运行`uv run --locked python scripts/check_publication.py --history`，不要把自动扫描当作图像或第三方许可人工审查的替代品。

DeepSeek直连回归：`uv run --locked pytest tests/providers/test_deepseek.py -q`；离线验证无效环境代理/证书覆盖不影响模型客户端、环境变量不被改写，以及普通/流式工具协议和关闭连接。自动测试不访问真实API，不消耗模型费用。

## 训练只读监测补充

`uv run --locked pytest tests/training/test_training_observe.py -q`验证进程/历史结果区分、日志有界分页、坏JSONL、来源时间、路径/硬链接/符号链接拒绝、仅查看模式查询和模型工具。测试只使用临时项目，不启动训练、不调用模型。完整训练模块测试：`uv run --locked pytest tests/training -q`；代码检查包括`uv run --locked ruff check src tests services/training_worker`。真实项目检查只采集已授权目录内日志/指标，不能以此替代训练或续训验收。`tests/manual/check_training_observe.py`在WSL临时目录启动一个仅sleep的假脚本，验证真实/proc关联、日志和指标读取，随后清理；不调用模型或GPU。
