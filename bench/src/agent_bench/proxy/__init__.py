"""OpenAI 兼容的本地录制代理。

只绑定 127.0.0.1；被测 agent 的 base_url 指向 `http://127.0.0.1:<port>/p/<scope>`，
由代理转发到真实上游，并把每次调用的 token、时延与状态记录下来。
"""

from agent_bench.proxy.recorder import ProxyRecorder
from agent_bench.proxy.server import RecordingProxy

__all__ = ["ProxyRecorder", "RecordingProxy"]
