"""仅校验启动目录，绝不声称限制子进程访问范围。"""

import hashlib
import shutil
from pathlib import Path

from lynx_agent.execution.models import ApprovalRequest
from lynx_agent.tools.errors import ToolExecutionError

SCRIPT_SUFFIXES = {".py", ".pyw", ".ps1", ".sh", ".bash", ".js", ".mjs", ".cjs", ".ts", ".rb", ".pl"}
MAX_SCRIPT_BYTES = 16 * 1024 * 1024


def script_contents(argv, directory):
    """绑定直接脚本参数的内容；不覆盖被导入模块或被脚本再读取的文件。"""
    found = []
    for raw in argv[1:]:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = Path(directory) / candidate
        if candidate.suffix.casefold() not in SCRIPT_SUFFIXES:
            continue
        try:
            info = candidate.stat()
        except OSError:
            continue
        if not info.st_size <= MAX_SCRIPT_BYTES:
            continue
        hasher = hashlib.sha256()
        try:
            with candidate.open("rb") as stream:
                while block := stream.read(65536):
                    hasher.update(block)
        except OSError:
            continue
        found.append((str(candidate.resolve()), hasher.hexdigest()))
    return tuple(sorted(found))


def prepare(arguments, policy, config, session, generation):
    if not policy.writable or not session:
        raise ToolExecutionError("当前权限或会话不允许执行命令")
    directory = policy.resolve(arguments.working_dir, write=True)
    if not directory.is_dir():
        raise ToolExecutionError("命令工作目录不存在或不是目录")
    executable = shutil.which(arguments.argv[0])
    if executable is None:
        raise ToolExecutionError("找不到程序，请先确认程序已安装并在 PATH 中；未执行")
    executable = Path(executable).resolve()
    if executable.suffix.lower() in {".cmd", ".bat"}:
        raise ToolExecutionError("第一版不执行批处理文件；请直接指定对应可执行程序")
    argv = (str(executable), *arguments.argv[1:])
    return ApprovalRequest.create(argv, directory, arguments.timeout_seconds or config.default_timeout_seconds,
                                  policy.mode, session, generation, script_contents(argv, directory))
