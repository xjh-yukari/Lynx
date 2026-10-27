import json
import re
from bisect import bisect_right
from pathlib import Path

from document_rag.config import digest


def source_units(path, fields=None):
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        section = None
        for part in re.split(r"(?m)(?=^#{1,6} )", path.read_text(encoding="utf-8-sig")):
            if part.strip():
                if part.startswith("#"):
                    section = part.splitlines()[0].lstrip("# ")[:512]
                yield part.strip(), None, section, path.name, ""
    elif suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(path)
        for number, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            if not text.strip():
                raise ValueError(f"第{number}页无可提取正文，需要OCR或显式排除文件")
            yield text.strip(), number, None, path.name, ""
    elif suffix in {".json", ".jsonl"}:
        if (not fields or not isinstance(fields.get("text"), list) or not fields["text"]
                or not all(isinstance(key, str) and key for key in fields["text"]) or not fields.get("id") or not fields.get("title")):
            raise ValueError("JSON 必须显式指定 text字段列表、title字段和id字段")
        raw = path.read_text(encoding="utf-8-sig")
        entries = [json.loads(line) for line in raw.splitlines() if line.strip()] if suffix == ".jsonl" else json.loads(raw)
        if not isinstance(entries, list):
            raise ValueError("JSON 顶层必须为数组")
        seen = set()
        for entry in entries:
            identifier = str(entry[fields["id"]])
            if identifier in seen:
                raise ValueError("JSON记录id重复")
            seen.add(identifier)
            text = "\n".join(str(entry[key]) for key in fields["text"] if entry.get(key) is not None)
            if not text.strip():
                raise ValueError("JSON记录无正文")
            yield (text, None, "record:" + identifier, str(entry.get(fields.get("title", ""), path.name))[:256],
                   str(entry.get(fields.get("asset", ""), "")))
    else:
        raise ValueError("不支持的文档类型")


def chunks(path, settings, encoder, fields=None):
    path = Path(path).resolve(strict=True)
    if not path.is_file() or path.stat().st_size > settings.max_file_bytes:
        raise ValueError("文件类型或大小超限")
    source = path.as_posix()
    doc_id = digest(source.casefold())
    file_hash = digest(path.read_bytes().hex())
    rows = []
    for text, page, section, title, asset_path in source_units(path, fields):
        # 用fast tokenizer偏移切原文，不能decode后把归一化文本冒充原文。
        encoded = encoder.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoded["offset_mapping"]
        ends = [stop for _, stop in offsets]
        start = 0
        while start < len(offsets):
            stop = min(len(offsets), start + settings.chunk_tokens)
            if stop < len(offsets):
                paragraph = text.rfind("\n\n", offsets[start][0], offsets[stop - 1][1])
                candidate = bisect_right(ends, paragraph) if paragraph >= 0 else 0
                if candidate - start >= settings.chunk_tokens // 2:
                    stop = candidate
            content = text[offsets[start][0]:offsets[stop - 1][1]].strip()
            if content:
                row = {"doc_id": doc_id, "content_hash": digest(content), "chunk_index": len(rows),
                       "text": content, "source": source, "title": title[:256], "section": section or "",
                       "page": page or 0, "asset_path": asset_path,
                       "record_id": section.removeprefix("record:") if section and section.startswith("record:") else ""}
                row["chunk_id"] = digest([doc_id, file_hash, row["chunk_index"], content, settings.manifest()])
                for key in ("text", "source", "title", "section", "asset_path", "record_id"):
                    if len(row[key].encode("utf-8")) > {"text": 16000, "source": 2048, "title": 1024, "section": 2048, "asset_path": 2048, "record_id": 512}[key]:
                        raise ValueError(f"字段UTF-8字节超限：{key}")
                rows.append(row)
            if stop >= len(offsets):
                break
            start = max(start + 1, stop - settings.overlap_tokens)
    if not rows:
        raise ValueError("文档无可提取正文")
    return rows, file_hash
