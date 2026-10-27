import argparse
import json

from document_rag.config import load
from document_rag.storage import Lease, Store


def main():
    parser = argparse.ArgumentParser(description="离线建库/发布/回滚；不会修改源文档或Engram")
    parser.add_argument("--config", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("build")
    create.add_argument("--plan", required=True)
    create.add_argument("--kb", default="default")
    create.add_argument("--exclude", action="append", default=[])
    for operation in ("publish", "rollback", "cleanup"):
        p = sub.add_parser(operation)
        p.add_argument("--collection", required=True)
        p.add_argument("--kb", default="default")
    settings = load(parser.parse_args().config)
    args = parser.parse_args()
    lease, store = Lease(settings.state_dir), None
    try:
        store = Store(settings)
        if args.command == "build":
            from document_rag.build import build
            from document_rag.encoding import Encoder
            result = build(settings, Encoder(settings), store, args.kb, args.plan, args.exclude)
        elif args.command in {"publish", "rollback"}:
            store.manifest(args.collection)
            store.client.load_collection(collection_name=args.collection, timeout=120)
            store.publish(args.kb, args.collection)
            result = store.manifest(args.collection)
        else:
            manifest = store.manifest(args.collection, compatible=False)
            if manifest["kb_id"] != args.kb or store.alias_names(args.collection):
                raise ValueError("禁止清理有alias或知识库归属不符的版本；先停止所有在线服务")
            store.client.drop_collection(collection_name=args.collection)
            result = {"removed": args.collection, "notice": "仅删除显式指定的旧索引，可由源文档重建"}
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        if store:
            store.close()
        lease.close()


if __name__ == "__main__":
    main()
