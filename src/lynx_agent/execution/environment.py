"""必要环境白名单及有界流式脱敏；无法替代隔离。"""

import codecs
import csv
import locale
import os
import re
import subprocess
from pathlib import Path

from pydantic import BaseModel, SecretStr

BASE_KEYS = ("SystemRoot", "SystemDrive", "windir", "ComSpec", "PATH", "PATHEXT", "TEMP", "TMP",
             "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "APPDATA", "LOCALAPPDATA", "PROGRAMDATA",
             "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "VIRTUAL_ENV", "HOME", "LANG", "TERM",
             "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")
SENSITIVE = re.compile(r"TOKEN|KEY|SECRET|PASSWORD", re.IGNORECASE)


def build_environment(extra=()):
    allowed = {key.upper() for key in (*BASE_KEYS, *extra)}
    env = {key: value for key, value in os.environ.items()
           if (key.upper() in allowed or key.upper().startswith("UV_")) and not SENSITIVE.search(key)}
    env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    return env


def protect_log_directory(path):
    """日志限制为当前账户；同一账户的获批命令仍可访问，不是沙箱。"""
    path.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        flags = {"creationflags": 0x08000000}
        identity = subprocess.check_output(["whoami", "/user", "/fo", "csv", "/nh"], text=True, **flags)
        sid = next(csv.reader(identity.strip().splitlines()))[1]
        subprocess.run(["icacls", str(path), "/inheritance:r", "/grant:r", f"*{sid}:(OI)(CI)(F)", "*S-1-5-18:(OI)(CI)(F)"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **flags)
    else:
        path.chmod(0o700)


def known_secrets(config):
    found = set()

    def collect(value):
        if isinstance(value, SecretStr):
            found.add(value.get_secret_value())
        elif isinstance(value, BaseModel):
            for name in type(value).model_fields:
                collect(getattr(value, name))
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                collect(item)

    collect(config)
    for server in config.mcp.servers.values():
        if server.token_env and os.environ.get(server.token_env):
            found.add(os.environ[server.token_env])
        if server.token_file:
            with Path(server.token_file).open("r", encoding="ascii") as stream:
                value = stream.read(4097).strip()
            if len(value) > 4096:
                raise ValueError("MCP密钥文件异常，未开启命令工具")
            found.add(value)
    return tuple(sorted((value for value in found if value), key=len, reverse=True))


def decode(data):
    for encoding in dict.fromkeys(("utf-8", locale.getpreferredencoding(False))):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            pass
    return data.decode(locale.getpreferredencoding(False), errors="replace")


class StreamDecoder:
    """保留跨读取块的UTF-8字符；失败后按本机编码容错。"""
    def __init__(self):
        self.decoder = codecs.getincrementaldecoder("utf-8")("strict")

    def feed(self, block, *, final=False):
        pending = self.decoder.getstate()[0]
        try:
            return self.decoder.decode(block, final=final)
        except UnicodeDecodeError:
            self.decoder = codecs.getincrementaldecoder(locale.getpreferredencoding(False))("replace")
            return self.decoder.decode(pending + block, final=final)


class Redactor:
    def __init__(self, secrets):
        self.secrets = secrets
        self.pending = ""
        self.hold = max((len(secret) for secret in secrets), default=1) - 1

    def clean(self, text):
        for secret in self.secrets:
            text = text.replace(secret, "[已隐藏]")
        return text

    def feed(self, text, *, final=False):
        self.pending += text
        stop = len(self.pending) if final else max(0, len(self.pending) - self.hold)
        # 不在密钥中间截断，跨读取块的密钥同样不会泄漏。
        for secret in self.secrets:
            start = self.pending.find(secret)
            while start != -1:
                if start < stop < start + len(secret):
                    stop = start
                start = self.pending.find(secret, start + 1)
        result, self.pending = self.pending[:stop], self.pending[stop:]
        return self.clean(result)
