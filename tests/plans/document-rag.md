# 文档RAG验证记录与后续验收

这是历史验证记录，不代表当前服务状态；当前部署与目录以 `docs/product-plan.md` 和 `tests/README.md` 为准。

日期：2026-10-23。目标项目：`E:\github\nanobot`。课程代码仅静态参考，没有执行课程脚本或修改其数据库。

## 已验证的层级

1. 自动策略/边界测试：HTTP只允许本机IP和认证配置、授权撤销、旧记忆编码兼容、来源字段、稳定chunk_id、JSON映射、失败不发布、离线/在线互斥、超时后实际推理槽位不提前释放、alias实际响应形状、不同编码版本的归属检查，以及完全文件权限不能修改授权文件。
2. 真实HTTP协议测试：独立本机MCP服务，未认证401、非法Origin403、白名单工具发现、结构化结果。此层检索器为替身。
3. 真实记忆测试：临时Engram库经真实stdio MCP保存带来源结论，撤权后拒绝get及自动召回。LLM提取为替身。
4. 真实检索组件：BGE-M3固定快照＋Milvus 2.5.14＋真实HTTP MCP。证据见 `tests/reports/rag-last-real-report.json`。
5. 真实本地Agent联调：`tests/knowledge/test_full_local.py`显式启用后已通过。RAG模型/数据库、HTTP MCP、Agent工具循环和临时Engram均为真实实现；只有LLM是脚本替身。覆盖无工具/仅记忆/仅文档/两者四条路径，自动来源绑定及撤权后的工具、快照和记忆限制。

真实检索组件已验证：

- csr_array/csr_matrix公开稀疏适配、空行dense退化。
- 两类故意高相关的越权记录（项目、知识库），dense/sparse/hybrid均不返回。
- 重复构建识别、全量更新排除删除文档、失败保留旧alias、真实alias回滚。
- alias在查询期间切换时，当前请求仍使用固定物理版本。
- 多个真实HTTP客户端复用同一编码器，只暴露search_knowledge。
- 服务退出不release/drop共享collection；独立三个容器重启后alias、数据和加载状态仍存在。

三道人工问题上dense与hybrid的Recall@5均为100%。这个小样本不能证明混合更好，也不能外推正式资料质量。报告的检索耗时不包括编码；模型加载耗时也不是单次查询延迟。

## 明确没有宣称完成

- 付费DeepSeek真实工具选择准确率、自然语言答案的引用准确率。
- 用户正式文档导入或默认配置启用。
- OCR、视觉检索、增量同步、向量语义长期记忆、远程多用户认证部署。
- 撤权后擦除旧会话正文、摘要、备份或已经发送给模型的内容。

## 复测

普通测试在Agent根目录执行 `uv run pytest tests/knowledge -q`。

真实组件需要独立验证数据库就绪。开发时创建的验证容器已停止但卷保留，可按顺序启动：

```powershell
docker start lynx-rag-validation-etcd-1 lynx-rag-validation-minio-1 lynx-rag-validation-milvus-1
cd E:\github\nanobot\services\document_rag
.venv\Scripts\python.exe ..\..\tests\manual\validate_rag_real.py
```

真实本地Agent联调（同state目录不能与建库同时运行）：

```powershell
cd E:\github\nanobot
$env:LYNX_RAG_REAL = '1'
uv run pytest tests/knowledge/test_full_local.py -q
```

普通测试默认skip此重型联调，而不是把缺数据库当通过。结束后可以停止上述三个验证容器，不要加删除卷参数。不要停止或修改课程的milvus-standalone/milvus-etcd/milvus-minio。
