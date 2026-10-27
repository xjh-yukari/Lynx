"""短请求WSL桥：隔离Python启动、JSON走stdin、有界输出，无Shell字符串。"""

import asyncio
import json
from pathlib import Path, PureWindowsPath

from lynx_agent.execution.environment import build_environment


class TrainingError(ValueError):
    pass


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise TrainingError("响应含重复JSON字段")
        result[key] = value
    return result


def decode_response(data):
    try:
        result = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_object,
                            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        if (not isinstance(result, dict) or set(result) != {"api_version", "ok", "data", "error"}
                or type(result["api_version"]) is not int or result["api_version"] != 1
                or type(result["ok"]) is not bool):
            raise ValueError
        if not result["ok"]:
            allowed = {"invalid_request", "path_denied", "not_found", "registry_unavailable", "unsupported"}
            code = result["error"] if result["error"] in allowed else "worker_error"
            raise TrainingError(f"训练基础服务拒绝请求：{code}；未启动训练")
        if result["error"] is not None or not isinstance(result["data"], dict):
            raise ValueError
        return result["data"]
    except (UnicodeError, ValueError, TypeError, KeyError, RecursionError) as error:
        if isinstance(error, TrainingError):
            raise
        raise TrainingError("训练服务响应无效；未把原始输出回传模型") from None


async def run_bounded(argv, payload, timeout):
    process = await asyncio.create_subprocess_exec(
        *argv, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE, env=build_environment(),
    )

    async def read(stream):
        chunks = []
        size = 0
        while chunk := await stream.read(8192):
            size += len(chunk)
            if size > 131072:
                raise TrainingError("训练控制响应超限")
            chunks.append(chunk)
        return b"".join(chunks)

    async def exchange():
        async def send():
            process.stdin.write(payload)
            await process.stdin.drain()
            process.stdin.close()
        sender = asyncio.create_task(send())
        stdout = asyncio.create_task(read(process.stdout))
        stderr = asyncio.create_task(read(process.stderr))
        tasks = (sender, stdout, stderr)
        try:
            await asyncio.gather(*tasks)
            await process.wait()
            if process.returncode:
                raise TrainingError("WSL控制命令失败；请检查发行版、Python和源码路径")
            return stdout.result()
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    try:
        async with asyncio.timeout(timeout):
            return await exchange()
    except TimeoutError:
        raise TrainingError("WSL请求超时；不会自动重试写入请求") from None
    finally:
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await process.wait()


class WSLClient:
    def __init__(self, config, *, invoke=run_bounded):
        self.config = config
        self.invoke = invoke
        self._bootstrap = None

    async def request(self, operation, arguments):
        prefix = ["wsl.exe", "--distribution", self.config.distribution, "--exec"]
        if self._bootstrap is None:
            source = Path(__file__).resolve().parents[3] / "services" / "training_worker" / "bootstrap.py"
            if not PureWindowsPath(str(source)).drive:
                raise TrainingError("此客户端用于Windows调用WSL；Linux测试请用worker独立入口")
            mapped = await self.invoke([*prefix, "/usr/bin/wslpath", "-u", str(source)], b"", self.config.rpc_timeout_seconds)
            path = mapped.decode("utf-8").strip()
            if not path.startswith("/") or "\n" in path or "\x00" in path:
                raise TrainingError("WSL源码映射无效")
            self._bootstrap = path
        payload = json.dumps({"api_version": 1, "operation": operation,
                              "root": self.config.root, "arguments": arguments}, allow_nan=False).encode()
        if len(payload) > 32768:
            raise TrainingError("训练控制参数过大")
        response = await self.invoke([*prefix, "/usr/bin/python3", "-I", self._bootstrap],
                                     payload, self.config.rpc_timeout_seconds)
        return decode_response(response)
