"""验证容器重启后的独立验证库；不创建/加载/重建/删除collection。"""
import json
import time
from pathlib import Path

from document_rag.config import load
from document_rag.storage import Store, atomic_json

directory = Path(__file__).resolve().parents[2]
settings = load(directory / "tests" / "fixtures" / "document_rag" / "settings.toml")
assert settings.uri == "http://127.0.0.1:19540" and settings.project_id == "rag_validation"
report_path = directory / "tests" / "reports" / "rag-last-real-report.json"
report = json.loads(report_path.read_text(encoding="utf-8"))
deadline = time.monotonic() + 90
while True:
    store = None
    try:
        store = Store(settings)
        physical, _ = store.resolve("default")
        assert physical == report["physical"]
        rows = store.client.query(collection_name=physical, filter='project_id == "rag_validation" and kb_id == "default"',
                                  output_fields=["source"], limit=10, timeout=10)
        assert len(rows) == 3
        report["checks"]["container_restart_persists_alias_data_and_load_state"] = True
        atomic_json(report_path, report)
        print("验证通过：独立容器重启后，alias、三份文档及加载状态仍存在；未执行load/drop。")
        break
    except Exception:
        if time.monotonic() >= deadline:
            raise
        time.sleep(2)
    finally:
        if store:
            store.close()
