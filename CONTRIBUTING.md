# 贡献指南

欢迎提交可复现的问题报告和范围清晰的改进。请先阅读 [产品方案](docs/product-plan.md)、[协作规则](AGENTS.md)、[安全说明](SECURITY.md) 和 [Apache-2.0 许可证](LICENSE)。请确保有权提交相关内容，并保留第三方要求的版权和许可声明。

1. 使用 Python 3.11+，在源码根目录运行 `uv sync --frozen`。本地 `.env` 仅保存自己的测试密钥，不上传。
2. 使用独立分支，只改本次任务相关文件，保留已有未提交修改。
3. 行为变更添加成功及失败边界测试；权限、持久化、工具执行需有回归用例。
4. 运行 `uv run --locked pytest -q`、`uv run --locked ruff check src tests services/training_worker`、`git diff --check`。跳过或未运行的集成验收需明确说明。
5. 所有测试使用临时数据；不擅自调用付费 API、重建知识库、操作真实实验或扩大权限来通过测试。
6. 运行 `uv run --locked python scripts/check_publication.py --history`，再人工审查差异。确认无密钥或私人数据后创建 commit。
7. PR 说明需求、实现、测试结果、限制和兼容性。勿上传本机配置、运行数据、含密钥的错误信息或大模型权重。

通用说明放 README / 产品方案；组件运维保留独立 README；测试入口放 `tests/README.md`，避免堆积重复或过时教程。
