"""专用下载工具：来源授权、逐次重定向校验、大小/超时限制，下载后不执行。"""

import asyncio
import hashlib
import json
import os
from urllib.parse import unquote

import aiohttp
from pydantic import Field
from yarl import URL

from lynx_agent.approval.models import PlanFile
from lynx_agent.tools.builtin.controlled import ControlledArguments, ControlledTool
from lynx_agent.tools.errors import ToolExecutionError
from lynx_agent.web.errors import UnsafeURLError
from lynx_agent.web.security import PublicResolver, public_url


class DownloadArguments(ControlledArguments):
    url: str = Field(min_length=1, max_length=4096)
    destination: str = Field(min_length=1, max_length=4096)
    filename: str = ""


class _Stopped(Exception):
    pass


def validate_url(raw, allow_private=False):
    if not allow_private:
        try:
            return public_url(raw)
        except UnsafeURLError as error:
            raise ToolExecutionError(str(error)) from None
    if len(raw) > 4096 or any(char.isspace() or ord(char) < 32 for char in raw) or "\\" in raw:
        raise ToolExecutionError("下载地址包含非法字符或长度超限")
    try:
        url = URL(raw)
        host = (url.raw_host or "").lower().rstrip(".")
        if url.scheme not in {"http", "https"} or not host:
            raise ToolExecutionError("只允许 HTTP/HTTPS 下载地址")
        if url.user is not None or url.password is not None:
            raise ToolExecutionError("不允许包含账号密码的下载地址")
        return url.with_fragment(None)
    except (ValueError, UnicodeError):
        raise ToolExecutionError("下载地址格式不合法") from None


def network_source(url):
    """主机和端口共同构成下载来源；重定向到不同端口也视为新来源。"""
    return f"{(url.raw_host or '').lower().rstrip('.')}:{url.port}"


class DownloadFileTool(ControlledTool):
    name = "download_file"
    description = ("从明确列出的 HTTPS/HTTP 来源下载单个文件到已授权输出目录；每次重定向重新校验目标主机，"
                   "来源主机必须出现在审批卡片中。限制大小、超时和并发，不发送 Cookie/凭据，不覆盖同名文件，"
                   "下载后绝不自动执行。需要用户审批。不是沙箱。")
    arguments_model = DownloadArguments

    def __init__(self, service, approval, *, allow_private=False):
        super().__init__(service, approval)
        self.allow_private = allow_private

    def _filename(self, url, supplied):
        name = supplied.strip() if supplied else unquote(url.path.rsplit("/", 1)[-1])
        if not name or name in {".", ".."} or any(char in name for char in '/\\:*?"<>|') or len(name) > 255:
            raise ToolExecutionError("下载文件名不合法；请显式提供单个安全文件名")
        if any(ord(char) < 32 for char in name) or name.endswith((" ", ".")):
            raise ToolExecutionError("下载文件名包含控制字符或尾随空格/点")
        return name

    def build_plan(self, arguments):
        url = validate_url(arguments["url"], self.allow_private)
        destination = self.candidate_directory(arguments["destination"])
        name = self._filename(url, arguments.get("filename", ""))
        target = destination / name
        self.policy.resolve(str(target), write=True, scoped_roots=(destination,))
        if target.exists():
            raise ToolExecutionError("下载目标已存在；不覆盖，请换一个文件名")
        host = (url.raw_host or "").lower().rstrip(".")
        source = network_source(url)
        plan = self.plan(
            tool=self.name, operation="download", purpose=arguments.get("purpose", ""),
            files=(PlanFile(str(target), 0, ""),), read_roots=(), write_roots={destination},
            network_sources=(source,),
            details=(("下载来源", f"{url.scheme}://{source}{url.path}"),
                     ("大小上限", f"{self.approval.config.download_max_bytes} 字节"),
                     ("重定向", "每次重新校验；出现新主机或端口必须重新授权"),
                     ("上传", "否；不发送本地文件或凭据"), ("执行", "下载后不自动运行")),
        )
        return plan, url, target, source, host

    async def execute(self, **arguments):
        parsed = self._parse(arguments)
        plan, url, target, source, host = self.build_plan(parsed)
        grant = await self.authorize(plan)
        write_roots = tuple(grant.plan.write_roots)
        allowed_sources = set(grant.plan.network_sources)
        target = self.policy.resolve(str(target), write=True, scoped_roots=write_roots)
        if target.exists():
            raise ToolExecutionError("下载目标已存在；不覆盖")
        limit = self.approval.config.download_max_bytes
        hasher = hashlib.sha256()
        size = 0
        try:
            async with asyncio.timeout(self.approval.config.download_timeout_seconds):
                connector = aiohttp.TCPConnector(
                    resolver=None if self.allow_private else PublicResolver(),
                    use_dns_cache=False, limit=1)
                timeout = aiohttp.ClientTimeout(total=self.approval.config.download_timeout_seconds)
                async with aiohttp.ClientSession(
                        connector=connector, trust_env=False, auto_decompress=False,
                        cookie_jar=aiohttp.DummyCookieJar(), timeout=timeout) as session:
                    current = url
                    for hop in range(self.approval.config.download_max_redirects + 1):
                        current = validate_url(str(current), self.allow_private)
                        if url.scheme == "https" and current.scheme != "https":
                            raise ToolExecutionError("不允许从 HTTPS 降级到 HTTP，下载已拒绝")
                        destination_source = network_source(current)
                        if destination_source not in allowed_sources:
                            raise ToolExecutionError("重定向到未授权来源；需重新申请下载来源授权")
                        headers = {"User-Agent": "LynxAgent/0.1 (controlled download)",
                                   "Accept": "*/*", "Accept-Encoding": "identity"}
                        async with session.get(current, headers=headers, allow_redirects=False) as response:
                            if response.status in {301, 302, 303, 307, 308}:
                                location = response.headers.get("Location")
                                if not location or hop >= self.approval.config.download_max_redirects:
                                    raise ToolExecutionError("重定向次数超过限制或被拒绝")
                                current = current.join(URL(location))
                                continue
                            if response.status >= 400:
                                raise ToolExecutionError(f"下载返回 HTTP {response.status}，未保存文件")
                            if response.content_length is not None and response.content_length > limit:
                                raise ToolExecutionError("下载内容超过大小限制，未保存文件")
                            with target.open("xb") as stream:
                                try:
                                    async for block in response.content.iter_chunked(65536):
                                        if self.approval.should_stop():
                                            raise _Stopped
                                        size += len(block)
                                        if size > limit:
                                            raise ToolExecutionError("下载内容超过大小限制，已删除未完成文件")
                                        hasher.update(block)
                                        stream.write(block)
                                    stream.flush()
                                    os.fsync(stream.fileno())
                                finally:
                                    stream.close()
                            break
                    else:
                        raise ToolExecutionError("重定向次数超过限制")
        except _Stopped:
            target.unlink(missing_ok=True)
            self.approval.record_operation(tool=self.name, authorization=grant.identifier,
                                           target=str(target), result="stopped", detail="下载被用户停止")
            result = json.loads(self.finish(
                plan, tool=self.name, status="stopped", succeeded=0, failed=0, skipped=0,
                changed=[], details=[{"path": str(target), "reason": "用户停止，未保存"}], stopped=True))
            result.update({"path": str(target), "url": str(url), "bytes": 0, "sha256": None,
                           "executed": False, "host": host, "source": source})
            return json.dumps(result, ensure_ascii=False)
        except ToolExecutionError:
            target.unlink(missing_ok=True)
            raise
        except (aiohttp.ClientError, TimeoutError, OSError) as error:
            target.unlink(missing_ok=True)
            raise ToolExecutionError(f"下载失败，未保存文件：{type(error).__name__}") from None
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        digest = hasher.hexdigest()
        self.approval.record_operation(tool=self.name, authorization=grant.identifier,
                                       target=str(target), result="downloaded",
                                       detail=f"{size} 字节 sha256={digest} source={source}")
        result = json.loads(self.finish(plan, tool=self.name, status="completed", succeeded=1, failed=0,
                                        skipped=0, changed=[str(target)], details=[]))
        result.update({"path": str(target), "url": str(url), "bytes": size, "sha256": digest,
                       "host": host, "source": source, "executed": False,
                       "notice": "下载完成但没有执行；文件内容不可信，运行前需单独的代码执行审批。"})
        return json.dumps(result, ensure_ascii=False)
