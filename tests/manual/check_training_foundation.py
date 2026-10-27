"""在Windows或WSL运行M1临时冒烟：不调用模型、不使用GPU、不启动训练。"""

import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4


def main():
    bootstrap = Path(__file__).resolve().parents[2] / "services" / "training_worker" / "bootstrap.py"
    with TemporaryDirectory(prefix="lynx-training-m1-") as temporary:
        root = Path(temporary)
        source = root / "source with space 中文"
        source.mkdir()
        marker = source / "should-not-exist"
        (source / "train.py").write_text(
            f"from trl import SFTTrainer, GRPOTrainer\nfrom pathlib import Path\nPath({str(marker)!r}).touch()\n"
            "a = SFTTrainer()\nb = GRPOTrainer()\n", encoding="utf-8")

        def request(operation, arguments):
            packet = {"api_version": 1, "root": str(root / "state"), "operation": operation, "arguments": arguments}
            result = subprocess.run([sys.executable, "-I", str(bootstrap)], input=json.dumps(packet).encode(),
                                    capture_output=True, check=True, timeout=15)
            response = json.loads(result.stdout)
            return response

        project = request("register", {"source": str(source), "distribution": "isolated-smoke"})["data"]
        discovered = request("inspect", {"project_id": project["id"]})["data"]
        assert discovered["framework_candidates"] == ["trl"]
        assert not marker.exists()
        for method, value in (("sft", 0.5), ("grpo", 0.6)):
            response = request("record", {"project_id": project["id"], "method": method, "protocol": "fake-eval",
                                           "evidence": "synthetic_unverified", "metrics": {"score": value},
                                           "experiment_id": str(uuid4())})
            assert response["ok"]
        records = request("experiments", {"project_id": project["id"]})["data"]["experiments"]
        assert len(records) == 2
        assert all(item["verification"] == "user_reported_unverified" for item in records)
        comparison = request("compare", {"project_id": project["id"], "experiment_ids": [row["id"] for row in records]})["data"]
        assert comparison["comparability"] == "not_verified"
        assert abs(comparison["candidate_minus_baseline"]["score"] - 0.1) < 1e-8
        assert request("plan", {"project_id": project["id"]})["data"]["executable"] is False
        assert request("submit_job", {})["error"] == "unsupported"
        print(json.dumps({"result": "passed", "checks": ["isolated_registration", "static_discovery_no_execution",
                          "sqlite_cross_process_persistence", "manual_metrics_marked_unverified", "planning_only",
                          "unverified_numeric_comparison", "training_submission_blocked"],
                          "gpu_used": False, "training_started": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
