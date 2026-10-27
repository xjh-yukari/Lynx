"""M1基础协议，仅台账/发现；拒绝提交、运行、暂停或取消训练。"""

import json
import sqlite3
import sys

from lynx_training_worker.discovery import environment, inspect, plan
from lynx_training_worker.observe import observe
from lynx_training_worker.registry import Registry


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("invalid_request")
        result[key] = value
    return result


def dispatch(packet):
    if (not isinstance(packet, dict) or set(packet) != {"api_version", "root", "operation", "arguments"}
            or type(packet["api_version"]) is not int or packet["api_version"] != 1
            or not isinstance(packet["arguments"], dict)):
        raise ValueError("invalid_request")
    operation, arguments = packet["operation"], packet["arguments"]
    observation = {"progress", "logs", "metrics", "artifacts"}
    if isinstance(operation, str) and operation in observation:
        required = {"project_id", "path"} if operation in {"logs", "metrics"} else {"project_id"}
        optional = {"offset", "limit"} if operation == "logs" else {"offset"} if operation == "artifacts" else set()
        if not required <= set(arguments) or set(arguments) - required - optional:
            raise ValueError("invalid_request")
        project = Registry(packet["root"]).project(arguments["project_id"])
        return observe(project, operation, arguments)
    expected = {"environment": set(), "projects": set(), "inspect": {"project_id"}, "plan": {"project_id"},
                "experiments": {"project_id"}, "compare": {"project_id", "experiment_ids"},
                "register": {"source", "distribution"},
                "record": {"project_id", "method", "protocol", "evidence", "metrics", "experiment_id"}}
    if not isinstance(operation, str) or operation not in expected:
        raise ValueError("unsupported")
    if set(arguments) != expected[operation]:
        raise ValueError("invalid_request")
    if operation == "environment":
        return environment()
    registry = Registry(packet["root"])
    if operation == "projects":
        return {"projects": registry.projects(), "phase": "M1", "scheduler_available": False}
    if operation == "register":
        return registry.register(**arguments)
    if operation == "record":
        return registry.record(**arguments)
    if operation == "experiments":
        return {"experiments": registry.experiments(**arguments)}
    if operation == "compare":
        return registry.compare(**arguments)
    project = registry.project(arguments["project_id"])
    discovered = inspect(project)
    return plan(project, discovered) if operation == "plan" else discovered


def main():
    try:
        raw = sys.stdin.buffer.read(32769)
        if len(raw) > 32768:
            raise ValueError("invalid_request")
        packet = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object,
                            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("invalid_request")))
        data = dispatch(packet)
        response = {"api_version": 1, "ok": True, "data": data, "error": None}
    except (ValueError, TypeError, KeyError, OSError, sqlite3.Error, RecursionError) as error:
        code = str(error) if type(error) is ValueError else "registry_unavailable"
        if code not in {"invalid_request", "path_denied", "not_found", "registry_unavailable", "unsupported"}:
            code = "invalid_request"
        response = {"api_version": 1, "ok": False, "data": None, "error": code}
    sys.stdout.write(json.dumps(response, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
