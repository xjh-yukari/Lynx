import math
from pathlib import Path

from document_rag.config import digest


def sparse_row(matrix, index):
    # csr_matrix 与 csr_array 都提供公开的二维切片和 tocoo。
    row = matrix[index:index + 1, :].tocoo()
    if row.shape[0] != 1:
        raise ValueError("稀疏向量必须为二维单行")
    result = {}
    for column, value in zip(row.col, row.data, strict=True):
        value = float(value)
        if not math.isfinite(value) or int(column) < 0:
            raise ValueError("非法稀疏向量")
        if value:
            result[int(column)] = result.get(int(column), 0.0) + value
    return result


class Encoder:
    def __init__(self, settings):
        from huggingface_hub import snapshot_download
        from pymilvus.model.hybrid import BGEM3EmbeddingFunction
        from transformers import AutoTokenizer

        self.settings = settings
        path = settings.model
        if not Path(path).is_dir():
            from huggingface_hub.errors import LocalEntryNotFoundError
            try:
                path = snapshot_download(path, revision=settings.model_revision, local_files_only=True)
            except LocalEntryNotFoundError:
                path = snapshot_download(path, revision=settings.model_revision,
                                         allow_patterns=["*.json", "*.bin", "*.pt", "*.model"])
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = BGEM3EmbeddingFunction(model_name=path, device=settings.device, use_fp16=False,
                                           normalize_embeddings=True, return_dense=True, return_sparse=True,
                                           return_colbert_vecs=False, batch_size=settings.batch_size,
                                           query_max_length=settings.model_max_tokens, passage_max_length=settings.model_max_tokens)
        # FlagEmbedding默认只编码512 tokens；必须显式扩展，不能把600-token块截断。
        if self.model.dim["dense"] != 1024:
            raise ValueError("模型维度必须为1024")
        self.fingerprint = digest(settings.manifest())

    def tokens(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def encode(self, texts, query=False):
        if any(self.tokens(text) + 2 > self.settings.model_max_tokens for text in texts):
            raise ValueError("输入超过模型上限，不允许静默截断")
        encoded = (self.model.encode_queries(texts) if query else self.model.encode_documents(texts))
        results = []
        for index, dense in enumerate(encoded["dense"]):
            dense = [float(value) for value in dense]
            if len(dense) != 1024 or any(not math.isfinite(value) for value in dense):
                raise ValueError("非法密集向量")
            norm = math.sqrt(sum(value * value for value in dense))
            if not 0.99 <= norm <= 1.01:
                raise ValueError("密集向量未归一化")
            results.append((dense, sparse_row(encoded["sparse"], index)))
        if len(results) != len(texts):
            raise ValueError("编码条数不匹配")
        return results
