"""命令审批绑定脚本内容：批准后脚本变化必须作废。"""

import sys
from uuid import uuid4

import pytest

from lynx_agent.execution.config import ExecutionConfig
from lynx_agent.execution.service import ExecutionService
from lynx_agent.filesystem.policy import FilePolicy
from lynx_agent.tools.errors import ToolExecutionError


@pytest.mark.asyncio
async def test_script_change_after_approval_invalidates_command(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    script = root / "task.py"
    script.write_text("print('first')", encoding="utf-8")
    session = str(uuid4())
    service = ExecutionService(FilePolicy(root), ExecutionConfig(), tmp_path / "logs", lambda: session)
    service.enabled = True

    def present(item):
        if item is not None:
            assert item.contents and item.contents[0][0].endswith("task.py")
            service.broker.decide(item.identifier, True)
            script.write_text("print('changed')", encoding="utf-8")

    service.broker.presenter = present
    with pytest.raises(ToolExecutionError, match="脚本内容已改变"):
        await service.execute({"argv": [sys.executable, "task.py"]})
    assert not list((tmp_path / "logs").rglob("*.log"))
