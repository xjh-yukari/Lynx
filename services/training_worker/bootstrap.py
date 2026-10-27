"""M1标准库控制入口；不安装环境、不启动训练、不运行项目模块。"""

import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
runpy.run_module("lynx_training_worker", run_name="__main__")
