# 训练助手 M1：发现与人工实验台账

已实现基础能力，不是完整训练服务。**不启动训练、不控制现有进程、不支持暂停续训**。没有安装systemd服务，不需要Torch或模型API，使用WSL系统Python标准库。Windows端依赖仓库内bootstrap源码路径，尚不是独立安装包。

已增加只读观察器：独立于通用文件工作区权限，已登记项目可在仅查看模式下重复查询，无需逐条Shell审批；首次注册仍需真实用户确认且不能在暂停/只读状态写台账。不扩大为任意磁盘读取、Shell执行或训练启停授权。

```text
/training progress <项目ID>
/training files <项目ID> [分页偏移]
/training logs <项目ID> run_grpo.log [距文件末尾字节偏移]
/training metrics <项目ID> outputs/grpo_run/summary.json
/training metrics <项目ID> outputs/grpo_run/metrics.jsonl
```

对应模型工具为training_progress、training_artifacts、training_logs、training_metrics，均read_only=True；自然语言提问可由DeepSeek调用、总结。斜杠命令不调用模型，断网也可查询WSL数据。先files找到路径，再logs/metrics；带空格路径加引号。logs默认最新8192字节，next_offset用于向前翻页（增长期间可能重叠）。files每页20条，扫描上限1000节点/4层，列表有截断标志，不保证发现整个项目。

观察器只读取授权项目内.log、stdout.txt、stderr.txt，以及config.json/summary.json/trainer_state.json/metrics.jsonl；模型权重只列元信息，不读正文，不读取数据集、隐藏路径或链接。指标只返回数值及完成标志，不回传原始配置、完整轨迹；JSONL窗口64KiB最多256行，坏行计数并跳过。每项附来源路径、文件时间与采集时间。仅做已知密钥与常见凭据格式脱敏，不承诺全面敏感信息识别。

Linux /proc通过脚本路径关联项目Python进程，返回PID及启动ticks，不回传完整命令/环境；进程存在不证明健康/进度，阶段仅为脚本名推测。模块启动(-m)、无法读取进程信息或无日志的终端打印可能无法识别，返回unknown/partial而不编造。日志、历史summary和活跃进程不自动关联具体实验，不因历史completed就宣布当前任务完成；日志中的Traceback也只是历史异常证据。监测按需采样，尚无持续轮询、自动告警或终端屏幕抓取。已有程序只向TTY打印且没有日志时，不能追溯恢复输出；后续启动请自行同步保存stdout/stderr。本功能不会重启现有程序来补日志。

## 在当前 Agent 中使用

重启Agent后：

```text
/training on confirm
/training environment
/training register /home/jiahaoxue/your-training-source confirm
/training projects
/training inspect <返回的项目ID>
/training plan <项目ID>
```

将源码示例路径替换为真实WSL绝对路径，带空格的路径用引号。不要照抄不存在的目录。注册授权只对该项目生效，不自动扫描主目录；环境和空列表查询不创建运行库。

人工登记与对比（数值仅举例，不是训练结果）：

```text
/training record <项目ID> sft eval-v1 accuracy=0.50 confirm
/training record <项目ID> grpo eval-v1 accuracy=0.60 confirm
/training records <项目ID>
/training compare <项目ID> <第一条实验ID> <第二条实验ID>
```

记录永远标为user_reported_unverified；共同协议名只是人工标签，不表示已核对数据/模型/预算。compare只计算共有指标差，不作效果或统计显著性结论。M1最近/列表分页能力有界：项目最多100条、人工实验最多50条；完整历史分页后续实现。显式两个ID的比较不受列表上限影响。

`/training off`仅关闭当前Agent工具，不删除台账，不停止任何已有训练。仅查看或审批paused模式禁止注册和登记。模型只有只读工具，不可自己注册、存指标或执行任意SQL。

## 不启动 Agent 的环境查询

在项目根目录执行，无需DeepSeek密钥：

```powershell
uv run --locked python -m lynx_agent.training environment
```

可用`--distribution`和`--root`显式指定目标；根目录默认`~/lynx-training`。不要把已有非空数据目录、系统目录或源码目录当台账根。实际配置只改示例，不自动覆盖用户config.toml；如需开机默认启用，人工合并config.example.toml的training段。

## 实现边界

- Windows客户端通过wslpath核实自身bootstrap映射，以argv调用隔离模式Python，JSON走stdin；控制响应有超时与大小上限，不回显原始stderr。
- 仅注册与人工登记写SQLite。M1短事务串行写，4张M1前缀表独立于技术稿的最终schema；未知数据库不覆盖。后续服务上线必须显式备份与迁移。
- 静态发现限制深度、节点、文件数量与字节数，跳过密钥、数据、模型与输出目录；不导入项目、不执行--help或train函数。发现结果是候选，不是依赖兼容或续训能力证明。
- 保护符号链接/重解析点/硬链接等特殊路径；这些是应用检查，不是抵抗同账户攻击的OS沙箱。
- 只有Windows→WSL端到端桥为当前正式目标；Linux原生使用bootstrap或人工冒烟入口，不声称已支持在Linux直接运行同款Windows客户端。
- submit/pause/resume/cancel明确拒绝，未创建任务。正式SFT/GRPO需提供源码完成框架、审批、调度和checkpoint适配。

## 测试

```powershell
uv run --locked pytest tests/training -q
uv run --locked ruff check src tests services/training_worker
uv run --locked python tests/manual/check_training_foundation.py
wsl.exe -d Ubuntu-24.04 --exec /usr/bin/python3 -I /mnt/e/github/nanobot/tests/manual/check_training_foundation.py
```

最后命令适用于当前仓库挂载位置，其他位置需先使用wslpath确认。冒烟只在临时目录创建假的训练源码与SQLite，假脚本不会被执行，不调用GPU、不调用模型，结束后清理。

完整设计见[技术设计](../../docs/training-assistant-technical-design.md)，后续真实训练验收见[接入计划](../../tests/plans/training-onboarding.md)。
