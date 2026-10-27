"""审批展示只接受宿主参数，过滤控制字符与已知密钥。"""

import json

from lynx_agent.cli.message_actions import clean_text
from lynx_agent.execution.environment import Redactor
from lynx_agent.execution.models import RISK


def approval_text(service, request):
    contents = ("".join(f"\n  {path} sha256={digest}" for path, digest in request.contents)
                if request.contents else "\n  （没有可识别的直接脚本参数）")
    text = (f"参数数组（按原顺序直接执行）：\n{json.dumps(request.argv, ensure_ascii=False, indent=2)}\n"
            f"工作目录：{request.working_dir}\n超时：{request.timeout_seconds}秒\n"
            f"脚本内容绑定：{contents}\n"
            f"权限：{request.permission_mode}\n会话：{request.session_id}\n{RISK}\n"
            "Y 执行一次 · N 拒绝 · Ctrl+C/Esc 拒绝并取消本轮；上下键浏览参数")
    return clean_text(Redactor(service.secrets).clean(text))
