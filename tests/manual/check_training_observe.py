"""WSL标准库冒烟：临时sleep假任务，不训练、不占GPU、不修改正式项目。"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "services" / "training_worker" / "src"))

from lynx_training_worker.__main__ import dispatch


def main():
    with tempfile.TemporaryDirectory(prefix="lynx-observe-") as temporary:
        base = Path(temporary)
        source = base / "source"
        source.mkdir()
        (source / "train_probe.py").write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
        (source / "run.log").write_text("step 3/10\n", encoding="utf-8")
        (source / "metrics.jsonl").write_text('{"step":3,"loss":0.5}\n', encoding="utf-8")

        def request(operation, arguments):
            return dispatch({"api_version": 1, "root": str(base / "state"),
                             "operation": operation, "arguments": arguments})

        project = request("register", {"source": str(source), "distribution": "isolated-smoke"})
        child = subprocess.Popen([sys.executable, str(source / "train_probe.py")], cwd=source)
        try:
            packet = {"project_id": project["id"]}
            progress = request("progress", packet)
            assert progress["current_state"] == "project_process_observed"
            assert any(p["pid"] == child.pid for p in progress["processes"]["matches"])
            assert request("metrics", {**packet, "path": "metrics.jsonl"})["latest_numeric_values"]["step"] == 3
            assert request("logs", {**packet, "path": "run.log"})["historical_progress_lines"] == ["step 3/10"]
            print(json.dumps({"temporary_project": True, "live_process_association": "passed",
                              "logs_and_metrics": "passed", "gpu_training": False}))
        finally:
            child.terminate()
            child.wait(timeout=5)
        assert request("progress", packet)["current_state"] != "project_process_observed"


if __name__ == "__main__":
    main()
