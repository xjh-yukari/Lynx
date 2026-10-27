"""不调用模型的M1命令入口；使用明确的 /training 同款控制命令。"""

import argparse
import asyncio

from lynx_agent.training.config import TrainingConfig
from lynx_agent.training.service import TrainingService


def main():
    parser = argparse.ArgumentParser(description="M1训练基础工具：不提交训练、不调用模型")
    parser.add_argument("--distribution", default="Ubuntu-24.04")
    parser.add_argument("--root", default="~/lynx-training")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    service = TrainingService(TrainingConfig(enabled=True, distribution=options.distribution, root=options.root),
                              writable=lambda: True)
    try:
        # argparse已分词，重新安全引用以支持带空格目录。
        import shlex
        print(asyncio.run(service.command(shlex.join(options.command))))
    except (ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
