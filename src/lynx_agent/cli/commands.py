"""终端多轮文本聊天入口。"""

import argparse
import asyncio

from lynx_agent.agent import AgentRunner
from lynx_agent.agent.errors import ContextError
from lynx_agent.agent.usage import UsageTracker
from lynx_agent.cli.formatting import (
    format_api_usage,
    format_diagnostics,
    format_tokens,
    format_tools,
)
from lynx_agent.cli.input import ChatInput
from lynx_agent.cli.permissions import (
    handle_approval_command,
    handle_approval_mode_command,
    handle_permissions_command,
)
from lynx_agent.cli.search import handle_search_command
from lynx_agent.cli.session_picker import NEW_SESSION, select_session
from lynx_agent.cli.status import OperationCancelled
from lynx_agent.config import ConfigError, load_config
from lynx_agent.providers import ProviderError
from lynx_agent.session import SessionError, SessionStore
from lynx_agent.tools.errors import ToolExecutionError

HELP_TEXT = """会话命令
  /new             准备新会话，发送首条消息后创建
  /sessions        打开历史会话选择
  /rename <名称>   修改当前会话名称（最长 80 字符）
  /clear           清空当前上下文，保留历史原文
  /save            保存当前会话
  /retry           恢复当前未完成答复，只整理已有结果，不重做工具
  /diagnostics     查看最近故障、生成预算与恢复动作（不会调用 API）
  /tokens [文本]   查看本地上下文估算及最近操作的 API 实际用量
  /cost           查看本次运行内当前会话的模型费用估算与峰谷说明
  /effort [status/off/low/high/max] TUI 中选择或查看推理强度（下次请求生效）
  /compact         手动压缩旧历史（调用模型，产生 API 费用）
  /tools           查看当前工具和执行限制（不会调用 API）
  /training        训练助手M1：只读监测、日志、指标、项目与实验台账；不提交训练
  /mcp [tools]     查看本地 MCP 连接或已注册工具
  /memory          查看长期记忆；list [offset] 查看已存内容，rules 查看保存规则；search/show/remember/forget/preview/approve/reject/pending/changes/undo/reconcile/recall on|off；autosave on|off 开关自动保留
  /profile         查看档案；show soul|user|agents；init；set 类别 内容；reload；changes
  /mcp connect <服务名> /mcp disconnect <服务名> 连接或断开已配置服务
  /search [on/off] 查看、开启或关闭关键词搜索（网页读取不变）
  /shell [status/on/off] 交互界面默认开启，每次执行按 Y/N 审批；关闭后可用 /shell on confirm 重开；不是沙箱
  /permissions [read/workspace/full] 查看或切换文件权限；完全权限需再次确认
  /approval [ask/auto/paused/revoke/stop] 查看或切换审批档位；默认常用操作自动，与文件权限独立
  /approvals       预览待审批的文件差异、敏感授权或批量任务
  /approve <编号> [batch] 确认操作；未预览时只展示预览，需再次输入确认
  /reject <编号/all> 拒绝待审批操作，不修改文件、不下载、不执行
  /help 或 help    显示帮助
  /exit 或 exit    退出聊天（quit 也可）

聊天输入按键
  Enter            发送整个输入框内容
  Alt+Enter        插入换行（也可先按 Esc 再按 Enter）
  Ctrl+C           清空当前草稿，继续输入
  Ctrl+D           退出聊天
  多行粘贴         保留换行，按 Enter 后才发送（需终端支持粘贴边界）
  只有单行输入会识别为命令，多行文本中的命令不会执行。

会话选择按键
  ↑ / ↓            移动选项
  Enter            进入选中会话
  Esc / Ctrl+C     取消选择，返回聊天
等待状态按键
  Ctrl+C           取消本次生成或压缩，清理完成后返回输入框
  取消等待不保证服务端停止计算，也不保证免除已发生的费用。
"""


def show_session(agent: AgentRunner, *, preview: bool = False) -> None:
    print(f"Lynx · {agent.session_title}")
    if agent.session_id is not None:
        print(f"会话 ID：{agent.session_id}")
    if preview:
        for message in agent.context.history[-4:]:
            if message.role == "tool" or message.tool_calls:
                continue
            role = "你" if message.role == "user" else "Lynx"
            print(f"{role}：{message.content}")


def show_api_usage(tracker: UsageTracker) -> None:
    """普通回复与摘要分别显示；未知用量不显示为零。"""
    print(format_api_usage(tracker))


async def chat(session_id: str | None = None) -> None:
    """读取输入、执行对话，并在退出时释放模型连接。"""
    config = load_config()
    store = SessionStore()
    runner = (
        AgentRunner.from_config(config)
        if session_id is None else AgentRunner.from_config(config, session_id=session_id)
    )
    async with runner as agent:
        reader = ChatInput()
        reader.execution = agent.execution
        reader.approval = agent.approval
        if agent.execution:
            from prompt_toolkit.application import get_app
            agent.execution.broker.presenter = lambda request: get_app().invalidate()
            try:
                agent.shell_command("on confirm")
            except (ValueError, OSError) as error:
                print(f"Shell 默认开启失败：{error}")
        if agent.approval is not None:
            from prompt_toolkit.application import get_app
            agent.approval.presenter = lambda pending: get_app().invalidate()
        show_session(agent, preview=session_id is not None)
        print("Enter 发送，Alt+Enter 换行；/help 查看帮助，/sessions 选择会话。")
        while True:
            try:
                user_input = await reader.read()
            except KeyboardInterrupt:
                print("已取消当前草稿。")
                continue
            except EOFError:
                break
            if not user_input.strip():
                continue
            # 仅单行识别为命令，防止粘贴资料误触 /clear、exit 等操作。
            command = user_input.strip() if "\n" not in user_input and "\r" not in user_input else ""
            if command.lower() in {"exit", "quit", "/exit"}:
                break
            if command.lower() in {"help", "/help"}:
                print(HELP_TEXT)
                continue
            parts = command.split(maxsplit=1)
            verb = parts[0] if parts else ""
            arguments = parts[1:]
            if verb in {"/memory", "/profile"}:
                try:
                    print(await agent.memory_command(verb, arguments[0] if arguments else ""))
                except (ValueError, ToolExecutionError) as error:
                    print(error)
                continue
            if verb == "/mcp":
                try:
                    print(await agent.mcp.command(arguments[0] if arguments else ""))
                except ToolExecutionError as error:
                    print(error)
                continue
            if verb == "/training":
                try:
                    print(await agent.training_command(arguments[0] if arguments else ""))
                except (ValueError, ToolExecutionError) as error:
                    print(error)
                continue
            if verb == "/tokens":
                pending = arguments[0] if arguments else None
                print(format_tokens(agent, config, pending))
                continue
            if verb == "/cost":
                print(agent.costs.details(agent.session_id))
                continue
            if command == "/tools":
                print(format_tools(agent, config))
                continue
            if command == "/diagnostics":
                print(format_diagnostics(agent))
                continue
            if verb == "/search":
                try:
                    print(handle_search_command(agent, arguments[0] if arguments else ""))
                except ValueError as error:
                    print(error)
                continue
            if verb == "/shell":
                try:
                    print(agent.shell_command(arguments[0] if arguments else ""))
                except (ValueError, OSError) as error:
                    print(error)
                continue
            if verb in {"/permissions", "/approvals", "/approve", "/reject", "/approval"}:
                try:
                    argument = arguments[0] if arguments else ""
                    print(handle_approval_mode_command(agent, argument) if verb == "/approval" else
                          handle_permissions_command(agent, argument) if verb == "/permissions" else
                          handle_approval_command(agent, verb, argument))
                except (ValueError, ToolExecutionError) as error:
                    print(error)
                continue
            if command == "/compact":
                try:
                    result = await reader.run_operation(
                        lambda status: agent.compact(on_status=status),
                    )
                except OperationCancelled:
                    print("已取消本次压缩，未提交的摘要已丢弃，原上下文保持不变。")
                except EOFError:
                    break
                except (ContextError, ProviderError, SessionError) as error:
                    print(f"压缩失败：{error}")
                else:
                    if result is None:
                        print("没有超出近期保留范围的旧历史，无需压缩。")
                    else:
                        print(f"压缩完成：约 {result.before_tokens:,} → {result.after_tokens:,} Tokens；摘要请求 {result.summary_calls} 次。")
                continue
            if command in {"/new", "/sessions"}:
                try:
                    selected = NEW_SESSION if command == "/new" else await select_session(store.list_sessions())
                    if selected is None:
                        continue
                    agent.switch_session(None if selected == NEW_SESSION else selected)
                except SessionError as error:
                    print(f"切换失败：{error}。仍停留在当前会话。")
                    continue
                show_session(agent, preview=True)
                continue
            if command == "/rename" or command.startswith("/rename "):
                try:
                    agent.rename_session(command.removeprefix("/rename").strip())
                except SessionError as error:
                    print(f"修改失败：{error}")
                else:
                    show_session(agent)
                continue
            if command == "/clear":
                try:
                    agent.clear_history()
                except SessionError as error:
                    print(f"清空失败：{error}")
                    continue
                print("对话历史已清空。")
                continue
            if command == "/save":
                try:
                    agent.save_session()
                except SessionError as error:
                    print(f"保存失败：{error}")
                else:
                    print("会话已保存。" if agent.session_id else "尚未创建会话，请先发送消息。")
                continue
            if command.startswith("/") and command != "/retry":
                print("未知命令，输入 /help 查看帮助。")
                continue
            previous_id = agent.session_id
            previous_compaction = agent.last_compaction
            try:
                response = await reader.run_stream_operation(
                    lambda status, text, message=user_input, retry=command == "/retry": agent.ask(
                        message, on_status=status, on_stream=text,
                    ) if not retry else agent.retry(on_status=status, on_stream=text),
                )
            except OperationCancelled:
                print("已取消本次请求，本轮消息未写入历史。")
                if agent.last_compaction is not None and agent.last_compaction is not previous_compaction:
                    print("本次请求中已完成的历史压缩已保留。")
                if agent.session_id != previous_id:
                    show_session(agent)
                continue
            except EOFError:
                break
            except ContextError as error:
                print(f"上下文处理失败：{error}")
                if agent.can_retry:
                    print("未完成任务已保留，使用 /diagnostics 查看；/retry 仅整理答复，不重做工具。")
                continue
            except (ValueError, ToolExecutionError) as error:
                print(f"无法继续：{error}")
                continue
            except SessionError as error:
                print(f"会话处理或保存失败：{error}。未提交完整回合，可能已有 API 消耗。")
                if agent.can_retry and agent.persistence_error:
                    print("未完成任务保留在内存；请先 /save 成功后再 /retry，勿重发写入任务。")
                continue
            except ProviderError as error:
                if agent.session_id != previous_id:
                    show_session(agent)
                print(f"调用失败：{error}。可以重新输入。")
                if agent.can_retry:
                    print("已有进度保留；/diagnostics 查原因，/retry 仅整理答复，不重做工具。")
                continue
            if agent.session_id != previous_id:
                show_session(agent)
            if agent.last_pruning is not None:
                print(f"已回收 {len(agent.last_pruning.replacements)} 条工具结果；估算变化可用 /tokens 查看。")
            if agent.last_compaction is not None:
                result = agent.last_compaction
                print(f"已自动压缩：约 {result.before_tokens:,} → {result.after_tokens:,} Tokens。")
            print(f"\nLynx：{response.content}")
            if agent.memory.retention_notice:
                print(agent.memory.retention_notice)
            if agent.persistence_error is not None:
                print(f"回答已生成，但保存失败：{agent.persistence_error}。可输入 /save 重试。")
            if response.finish_reason == "length":
                print("（未完整生成：自动重写已停止，本轮未加入上下文。草稿已记录，请拆分任务或提高上限。）")
                if agent.can_retry:
                    print("未完成任务已保存；/diagnostics 查看原因，/retry 仅整理已有结果。")
    print("已退出聊天。")


async def tui_chat(session_id: str | None = None) -> None:
    """默认启动统一 TUI；纯终端模式通过 --plain 保留。"""
    from lynx_agent.cli.app import LynxTUI

    config = load_config()
    store = SessionStore()
    async with AgentRunner.from_config(config, session_id=session_id, store=store) as runner:
        await LynxTUI(runner, config, store).run()


def main() -> None:
    """启动异步聊天，处理配置错误和用户中断。"""
    parser = argparse.ArgumentParser(description="Lynx 终端聊天")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--session", metavar="ID", help="恢复指定会话")
    group.add_argument("--list-sessions", action="store_true", help="列出已保存会话")
    parser.add_argument("--plain", action="store_true", help="使用逐行聊天界面，兼容旧终端")
    parser.add_argument(
        "--headless", nargs="?", const="", default=None, metavar="任务文本",
        help="非交互运行单个任务后退出；省略文本或传 - 时从 stdin 读取",
    )
    parser.add_argument("--json", action="store_true", help="headless 模式下在 stdout 输出 NDJSON 事件")
    parser.add_argument("task", nargs="?", default=None, help="headless 任务文本（也可用 --headless 后置参数）")
    args = parser.parse_args()
    if args.json and args.headless is None:
        parser.error("--json 只能与 --headless 一起使用")
    try:
        if args.list_sessions:
            sessions = SessionStore().list_sessions()
            for session in sessions:
                print(f"{session.id}  {session.display_title}  {session.updated_at.isoformat()}")
            if not sessions:
                print("暂无已保存会话。")
            return
        if args.headless is not None:
            from lynx_agent.cli.headless import run_cli

            task = args.headless or args.task or ""
            code = run_cli(None if task.strip() in {"", "-"} else task, args.session, json_mode=args.json)
            raise SystemExit(code)
        asyncio.run(chat(args.session) if args.plain else tui_chat(args.session))
    except (ConfigError, ProviderError, SessionError) as error:
        print(f"运行失败：{error}")
        raise SystemExit(1) from error
    except KeyboardInterrupt:
        print("\n已退出聊天。")
