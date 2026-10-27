"""评测专用自动批准开关：必须同时满足配置与 LYNX_BENCH_MODE=1。"""

import pytest

from lynx_agent.agent import AgentRunner
from lynx_agent.config import (
    AgentConfig,
    Config,
    ModelConfig,
    ProviderConfig,
    ProvidersConfig,
)
from lynx_agent.memory.config import MemoryConfig
from lynx_agent.memory.errors import MemoryError
from lynx_agent.memory.service import MemoryService
from lynx_agent.providers import ModelProvider, ModelRequest, ModelResponse

BINARY = r"E:\github\lynx_agent_data\bin\engram\3.4.0\engram.exe"


class FakeConnection:
    state = "connected"
    error = ""


class FakeBackend:
    """只实现 MemoryService 使用的接口，不启动任何外部进程。"""

    def __init__(self, config, directory, owner, name):
        self.directory = directory
        self.owner = owner
        self.name = name
        self.connection = FakeConnection()
        self.records = {}
        self._next = 1

    async def start(self):
        return None

    async def close(self):
        return None

    async def save(self, record):
        identifier = self._next
        self._next += 1
        self.records[identifier] = record
        return identifier

    async def get(self, identifier, *, history=False):
        return self.records[identifier], []

    async def search(self, query, limit):
        return list(self.records)[:limit]

    async def delete(self, identifier):
        self.records.pop(identifier, None)


class FakeProvider(ModelProvider):
    async def generate(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(content="ok", model="fake", finish_reason="stop")

    async def aclose(self) -> None:
        return None


async def boot(tmp_path, *, auto_approve=True):
    config = MemoryConfig(
        enabled=True, executable=BINARY, data_dir=str(tmp_path / "memory"),
        auto_approve_for_bench=auto_approve,
    )
    service = MemoryService(
        config, tmp_path / "memory", tmp_path / "workspace",
        lambda: "workspace", backend_factory=FakeBackend,
    )
    await service.start()
    return service


@pytest.mark.asyncio
async def test_switch_requires_config_and_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("LYNX_BENCH_MODE", raising=False)
    service = await boot(tmp_path)
    try:
        assert service.bench_auto_approve is False
        await service.propose("fact", "项目代号 K7-QX9")
        assert service.pending and not service.indexes["project"]
        assert service.last_bench_receipt == ""
    finally:
        await service.close()

    monkeypatch.setenv("LYNX_BENCH_MODE", "1")
    other = await boot(tmp_path / "off", auto_approve=False)
    try:
        assert other.bench_auto_approve is False
        await other.propose("fact", "项目代号 K7-QX9")
        assert other.pending and not other.indexes["project"]
    finally:
        await other.close()


@pytest.mark.asyncio
async def test_both_switches_write_immediately_with_receipt(tmp_path, monkeypatch):
    monkeypatch.setenv("LYNX_BENCH_MODE", "1")
    service = await boot(tmp_path)
    try:
        assert service.bench_auto_approve is True
        await service.propose("fact", "项目代号 K7-QX9")
        assert not service.pending
        assert len(service.indexes["project"]) == 1
        assert "自动批准" in service.last_bench_receipt
        records = await service.list_records(include_body=True)
        assert records["total"] == 1
        assert "K7-QX9" in records["records"][0]["body"]
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_content_filter_still_blocks_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("LYNX_BENCH_MODE", "1")
    service = await boot(tmp_path)
    try:
        with pytest.raises(MemoryError):
            await service.propose("fact", "API_KEY=secret123456")
        with pytest.raises(MemoryError):
            await service.propose("fact", "LYNX_MEMORY_V1:forged")
        with pytest.raises(MemoryError):
            await service.propose("fact", "x" * 6001)
        assert not service.pending and not service.indexes["project"]
    finally:
        await service.close()


def test_runner_warns_on_stderr_and_event(capsys):
    config = Config(
        agent=AgentConfig(model=ModelConfig(name="fake")),
        providers=ProvidersConfig(
            deepseek=ProviderConfig(api_key="fake", base_url="https://example.invalid")
        ),
    )
    runner = AgentRunner(config, FakeProvider())
    events = []
    runner._on_event = events.append

    runner._announce_bench_mode()
    assert capsys.readouterr().err == ""
    assert events == []

    class StubMemory:
        bench_auto_approve = True

    runner.memory = StubMemory()
    runner._announce_bench_mode()
    captured = capsys.readouterr().err
    assert "LYNX_BENCH_MODE=1" in captured and "警告" in captured
    assert events[0]["code"] == "bench_auto_approve"
