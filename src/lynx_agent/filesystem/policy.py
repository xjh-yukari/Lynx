"""应用级文件边界；不是操作系统沙箱。"""

import os
import stat
from pathlib import Path

from lynx_agent.tools.errors import ToolExecutionError

WORKSPACE = Path("E:/github/nanobot")
MODES = {"read": "仅查看", "workspace": "工作区修改", "full": "完全权限"}
EXCLUDED = {".git", ".venv", "venv", "node_modules", "__pycache__", "build", "dist", ".idea"}
PRIVATE_DIRS = {".ssh", ".aws", ".azure", ".gnupg", ".kube"}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".jsonl", ".toml", ".yaml", ".yml", ".ini", ".cfg",
                 ".csv", ".tsv", ".log", ".js", ".ts", ".tsx", ".jsx", ".html", ".css", ".sql", ".xml",
                 ".sh", ".ps1", ".bat", ".rs", ".go", ".c", ".h", ".cpp", ".java", ".gitignore", ".env",
                 ".pem", ".key"}


def _system_roots():
    names = ("SystemRoot", "ProgramFiles", "ProgramFiles(x86)", "ProgramData")
    values = [os.environ.get(name) for name in names]
    if os.name != "nt":
        values.extend(("/etc", "/usr", "/bin", "/sbin", "/boot", "/var/lib"))
    roots = []
    for value in values:
        if not value:
            continue
        try:
            roots.append(Path(value).resolve(strict=False))
        except OSError:
            continue
    return tuple(dict.fromkeys(roots))


SYSTEM_ROOTS = _system_roots()
AGENT_RULE_PATHS = tuple(
    (WORKSPACE / name).resolve(strict=False)
    for name in ("config.toml", "AGENTS.md", "SOUL.md", "USER.md",
                 "src/lynx_agent/approval", "src/lynx_agent/execution",
                 "src/lynx_agent/filesystem/policy.py")
)


class SensitivePath(ToolExecutionError):
    def __init__(self, path):
        self.path = path
        super().__init__("敏感文件需要用户单独确认")


class FilePolicy:
    def __init__(self, root=WORKSPACE):
        self.root = Path(root).absolute()
        self.mode = "workspace"
        self.sensitive_grants = set()
        # 由 Runner 注入当前会话存储目录，防止通用文件工具绕过存档权限。
        self.private_roots = set()

    @property
    def label(self):
        return MODES[self.mode]

    @property
    def writable(self):
        return self.mode != "read"

    def set_mode(self, mode):
        if mode not in MODES:
            raise ValueError("权限模式必须为 read、workspace 或 full")
        self.mode = mode
        self.sensitive_grants.clear()

    @staticmethod
    def sensitive(path):
        parts = [part.casefold() for part in path.parts]
        name = path.name.casefold()
        return (any(part in PRIVATE_DIRS for part in parts) or name.startswith((".env", "id_rsa", "id_ed25519", "id_ecdsa")) or
                name in {".netrc", "_netrc", ".npmrc", ".pypirc", ".git-credentials", "credentials", "credentials.json",
                         "credentials.toml", "secrets.json", "secrets.toml", "secrets.yaml"} or
                path.suffix.casefold() in {".pem", ".key", ".p12", ".pfx"})

    def resolve(self, raw, *, write=False, allow_sensitive=False, scoped_roots=()):
        return self._resolve(raw, write=write, allow_sensitive=allow_sensitive,
                             scoped_roots=scoped_roots, enforce_scope=True)

    def resolve_candidate(self, raw, *, allow_sensitive=False):
        """解析批量任务候选路径：允许工作区外，但仍执行全部链接/设备/私有目录检查。

        调用方必须先把候选目录写进可审阅的批量授权卡片；这里不授予任何执行权。
        """
        return self._resolve(raw, write=False, allow_sensitive=allow_sensitive,
                             scoped_roots=(), enforce_scope=False)

    def _resolve(self, raw, *, write, allow_sensitive, scoped_roots, enforce_scope):
        if not isinstance(raw, str) or not raw.strip() or len(raw) > 4096 or any(ord(c) < 32 for c in raw):
            raise ToolExecutionError("文件路径为空或格式不合法")
        if raw.startswith(("\\\\", "//")):
            raise ToolExecutionError("不支持网络共享或设备路径")
        supplied = Path(raw)
        devices = {"con", "prn", "aux", "nul", "conin$", "conout$", *[f"com{i}" for i in range(1, 10)], *[f"lpt{i}" for i in range(1, 10)]}
        if any(part.endswith((" ", ".")) or part.split(".")[0].casefold() in devices
               for part in supplied.parts if part not in {".", supplied.anchor}):
            raise ToolExecutionError("不支持 Windows 设备名或带尾随空格/点的路径")
        if any(":" in part for part in supplied.parts[1:]) or (not supplied.drive and ":" in raw):
            raise ToolExecutionError("不支持备用数据流或特殊路径")
        if ".." in supplied.parts or (supplied.drive and not supplied.is_absolute()):
            raise ToolExecutionError("不支持上级路径或驱动器相对路径")
        path = supplied if supplied.is_absolute() else self.root / supplied
        if write and not self.writable:
            raise ToolExecutionError("当前权限为仅查看，禁止写入")
        # 不跟随任何符号链接或 Windows 重解析点，包含目录联接。
        for part in [*reversed(path.parents), path]:
            try:
                info = part.lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise ToolExecutionError("不允许通过符号链接或目录联接访问文件")
        path = path.resolve(strict=False)
        if any(path.is_relative_to(private) for private in self.private_roots):
            raise ToolExecutionError("会话存档目录或受保护配置禁止通过通用文件工具访问；存档请按结果编号使用受限恢复工具")
        if any(path.is_relative_to(root) for root in SYSTEM_ROOTS):
            raise ToolExecutionError("系统目录属于保护范围，拒绝通过通用文件工具访问")
        if write and any(path.is_relative_to(root) for root in AGENT_RULE_PATHS):
            raise ToolExecutionError("Agent 权限、审批规则和持久指令属于保护范围，禁止通过工具修改")
        root = self.root.resolve(strict=False)
        scopes = tuple(Path(extra).resolve(strict=False) for extra in scoped_roots)
        if (enforce_scope and self.mode != "full" and not path.is_relative_to(root)
                and not any(path.is_relative_to(extra) for extra in scopes)):
            raise ToolExecutionError("路径超出固定工作区，当前权限禁止访问")
        if any(part.casefold() in EXCLUDED for part in path.parts):
            raise ToolExecutionError("该目录已排除，不允许读取或修改")
        if path.exists():
            info = path.stat()
            if not stat.S_ISREG(info.st_mode) and not stat.S_ISDIR(info.st_mode):
                raise ToolExecutionError("只支持普通本地文件和目录")
            if stat.S_ISREG(info.st_mode) and info.st_nlink > 1:
                raise ToolExecutionError("不支持硬链接文件")
        if self.sensitive(path) and str(path) not in self.sensitive_grants and not allow_sensitive:
            raise SensitivePath(path)
        return path

    @staticmethod
    def text_file(path):
        if path.suffix.casefold() not in TEXT_SUFFIXES and not FilePolicy.sensitive(path) and path.name.casefold() not in {
            "readme", "license", "dockerfile", "makefile", ".gitignore", ".python-version", ".env", "credentials",
        }:
            raise ToolExecutionError("第一版仅支持常见文本、代码与配置文件，不支持二进制、PDF 或 Office 文件")
