"""用于人工确认的可读差异，不执行写入。"""

import difflib


def preview_change(item, previous):
    updated = item.record.body if item.action == "save" else ""
    diff = "\n".join(
        difflib.unified_diff(
            previous.splitlines(),
            updated.splitlines(),
            fromfile="当前有效内容",
            tofile="拟修改内容",
            lineterm="",
        )
    )
    return f"{item.action} · {item.record.kind} · {item.key}\n{diff}\n确认：/memory approve {item.id}\n拒绝：/memory reject {item.id}"
