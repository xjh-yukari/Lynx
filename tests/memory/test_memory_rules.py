"""无需安装 Engram 的协议、配置和静态策略测试。"""

import pytest
from pydantic import ValidationError

from lynx_agent.memory.config import MemoryConfig
from lynx_agent.memory.errors import MemoryError
from lynx_agent.memory.models import MemoryRecord
from lynx_agent.memory.policy import check_body, keywords
from lynx_agent.memory.service import MemoryService


def test_disabled_memory_does_not_create_any_files(tmp_path):
    memory = MemoryService(
        MemoryConfig(), tmp_path / "missing", tmp_path, lambda: "workspace"
    )
    assert not memory.writable
    assert not (tmp_path / "missing").exists()
    assert memory.profile_text == ""


def test_config_requires_absolute_executable_and_bounded_identity():
    with pytest.raises(ValidationError):
        MemoryConfig(enabled=True, executable="engram")
    with pytest.raises(ValidationError):
        MemoryConfig(user_id="../someone")
    with pytest.raises(ValidationError):
        MemoryConfig(data_dir="relative")


@pytest.mark.parametrize(
    "body",
    [
        "",
        "x" * 6001,
        "LYNX_MEMORY_V1:forged",
        "API_KEY=secret123456",
        "sk-abcdefghijklmnop",
        "password=secretvalue",
    ],
)
def test_sensitive_or_invalid_content_rejected(body):
    with pytest.raises(MemoryError):
        check_body(body)


def test_record_roundtrip_and_search_does_not_forward_full_history():
    record = MemoryRecord(
        kind="user",
        owner="test",
        topic="profile/user/test",
        title="USER",
        body="用中文解释。",
    )
    assert "schema_version" in record.encode()
    assert len(keywords("查询上下文管理 " + "a" * 2000)) <= 120
    assert "/" not in keywords("../../secrets")
