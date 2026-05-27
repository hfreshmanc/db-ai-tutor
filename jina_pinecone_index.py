"""
Jina Embeddings ↔ Pinecone（与 test_jina_embeddings.py 同链路）。
建库、检索、清空均在此模块完成，不依赖 google.generativeai。
"""
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

METADATA_TEXT_LIMIT = 8000


@dataclass
class ChildChunkRecord:
    id: str
    child_text: str
    parent_text: str
    parent_id: str
    filename: str
    child_index: int


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = _env(name, str(default))
    try:
        return max(minimum, int(raw))
    except ValueError:
        return default


def _parse_dimensions() -> Optional[int]:
    dim_raw = _env("JINA_EMBEDDING_DIMENSIONS")
    if not dim_raw:
        return None
    try:
        return int(dim_raw)
    except ValueError:
        return None


def _split_by_length(text: str, size: int, overlap: int) -> List[str]:
    result: List[str] = []
    start = 0
    step = max(1, size - overlap)
    while start < len(text):
        chunk = text[start : start + size]
        if chunk.strip():
            result.append(chunk)
        start += step
    return result


def _chunk_strategy() -> str:
    strategy = _env("RAG_CHUNK_STRATEGY", "parent_child").lower()
    if strategy in ("parent_child", "parent-child", "parentchild"):
        return "parent_child"
    if strategy in ("fixed", "length", "simple"):
        return "fixed"
    return "parent_child"


def _parent_child_records(text: str, filename: str, run_id: str) -> List[ChildChunkRecord]:
    """Parent-Child：parent 保留完整知识点，child 用于向量检索。"""
    parent_size = _env_int("RAG_PARENT_SIZE", 1000, 100)
    parent_overlap = _env_int("RAG_PARENT_OVERLAP", 200, 0)
    child_size = _env_int("RAG_CHILD_SIZE", 250, 50)
    child_overlap = _env_int("RAG_CHILD_OVERLAP", 50, 0)

    parent_texts = _split_by_length(text, parent_size, parent_overlap) or [text]
    records: List[ChildChunkRecord] = []

    for p_idx, parent_text in enumerate(parent_texts):
        parent_id = f"p-{run_id}-{p_idx}"
        child_texts = _split_by_length(parent_text, child_size, child_overlap)
        if not child_texts:
            child_texts = [parent_text]

        for c_idx, child_text in enumerate(child_texts):
            if not child_text.strip():
                continue
            records.append(
                ChildChunkRecord(
                    id=f"c-{parent_id}-{c_idx}",
                    child_text=child_text.strip(),
                    parent_text=parent_text.strip(),
                    parent_id=parent_id,
                    filename=filename,
                    child_index=c_idx,
                )
            )
    return records


def _fixed_length_records(text: str, filename: str, run_id: str) -> List[ChildChunkRecord]:
    chunk_size = _env_int("RAG_CHUNK_SIZE", 800, 100)
    chunk_overlap = _env_int("RAG_CHUNK_OVERLAP", 80, 0)
    parts = _split_by_length(text, chunk_size, chunk_overlap) or [text]

    records: List[ChildChunkRecord] = []
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        parent_id = f"f-{run_id}-{i}"
        records.append(
            ChildChunkRecord(
                id=f"rag-{run_id}-{i}",
                child_text=part,
                parent_text=part,
                parent_id=parent_id,
                filename=filename,
                child_index=0,
            )
        )
    return records


def build_index_records(text: str, filename: str) -> Tuple[str, List[ChildChunkRecord]]:
    run_id = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
    strategy = _chunk_strategy()
    if strategy == "parent_child":
        records = _parent_child_records(text, filename, run_id)
    else:
        records = _fixed_length_records(text, filename, run_id)
    return strategy, records


def _truncate_metadata(value: str, limit: int = METADATA_TEXT_LIMIT) -> str:
    value = (value or "").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def get_chunk_strategy() -> str:
    return _chunk_strategy()


def _jina_embed(texts: List[str], task: str) -> List[List[float]]:
    """与 test_jina_embeddings.py 相同的 POST 体与 headers。"""
    api_key = _env("JINA_API_KEY")
    if not api_key:
        raise ValueError("JINA_API_KEY 未配置")

    url = _env("JINA_EMBEDDINGS_URL", "https://api.jina.ai/v1/embeddings")
    model = _env("JINA_EMBEDDING_MODEL", "jina-embeddings-v3")
    timeout = float(_env("JINA_HTTP_TIMEOUT_SECONDS", "300") or "300")
    dimensions = _parse_dimensions()

    body: Dict[str, Any] = {
        "model": model,
        "input": texts,
        "task": task,
    }
    if dimensions is not None:
        body["dimensions"] = dimensions

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    resp = requests.post(url, headers=headers, json=body, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"Jina HTTP {resp.status_code}: {resp.text}")

    rows = resp.json().get("data") or []
    rows = sorted(rows, key=lambda x: x.get("index", 0))
    if len(rows) != len(texts):
        raise RuntimeError(f"Jina 返回 {len(rows)} 条，期望 {len(texts)} 条")

    out: List[List[float]] = []
    for row in rows:
        emb = row["embedding"]
        if dimensions is not None and len(emb) != dimensions:
            raise RuntimeError(
                f"向量维度 {len(emb)} 与 JINA_EMBEDDING_DIMENSIONS={dimensions} 不一致"
            )
        out.append(emb)
    return out


def _pinecone_index():
    pinecone_key = _env("PINECONE_API_KEY")
    index_name = _env("PINECONE_INDEX")
    if not pinecone_key or not index_name:
        raise ValueError("请配置 PINECONE_API_KEY 与 PINECONE_INDEX")
    pc = Pinecone(api_key=pinecone_key)
    return pc.Index(index_name), index_name


def index_text_to_pinecone(text: str, filename: str) -> int:
    """上传文本 → Parent-Child 切块 → Jina(child, passage) → Pinecone upsert。"""
    text = (text or "").strip()
    if not text:
        return 0

    embed_batch = _env_int("JINA_EMBED_BATCH", 64, 1)
    model = _env("JINA_EMBEDDING_MODEL", "jina-embeddings-v3")
    task = _env("JINA_TASK_PASSAGE", "retrieval.passage")

    strategy, records = build_index_records(text, filename)
    if not records:
        return 0

    child_texts = [r.child_text for r in records]
    print(
        f"[jina→pinecone] index {filename!r}: strategy={strategy}, "
        f"{len(records)} child chunk(s) from {len({r.parent_id for r in records})} parent(s)"
    )

    all_embeddings: List[List[float]] = []
    for i in range(0, len(child_texts), embed_batch):
        batch_texts = child_texts[i : i + embed_batch]
        all_embeddings.extend(_jina_embed(batch_texts, task))

    vectors = []
    for record, emb in zip(records, all_embeddings):
        vectors.append(
            {
                "id": record.id,
                "values": emb,
                "metadata": {
                    "text": _truncate_metadata(record.child_text),
                    "parentContext": _truncate_metadata(record.parent_text),
                    "parentId": record.parent_id,
                    "filename": record.filename,
                    "chunkType": "child",
                    "chunkStrategy": strategy,
                    "childIndex": record.child_index,
                    "source": "api/rag/index",
                    "model": model,
                },
            }
        )

    index, index_name = _pinecone_index()
    upserted = 0
    for j in range(0, len(vectors), 100):
        batch_v = vectors[j : j + 100]
        res = index.upsert(vectors=batch_v)
        upserted += getattr(res, "upserted_count", None) or len(batch_v)

    print(f"[jina→pinecone] upserted {upserted} into {index_name!r}")
    return upserted


def _context_from_metadata(meta: Dict[str, Any]) -> Tuple[str, str]:
    child_text = (meta.get("text") or "").strip()
    parent_context = (meta.get("parentContext") or "").strip()
    context_text = parent_context or child_text
    return context_text, child_text


def search_pinecone(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """RAG 检索：Jina(query) → Pinecone query；命中 child，返回 parent context。"""
    q = (query or "").strip()
    if not q:
        return []

    task = _env("JINA_TASK_QUERY", "retrieval.query")
    vector = _jina_embed([q], task)[0]
    index, _ = _pinecone_index()

    # 多取一些候选，便于按 parent 去重后仍凑满 top_k
    fetch_k = max(top_k * 3, top_k)
    res = index.query(vector=vector, top_k=fetch_k, include_metadata=True)

    deduped: Dict[str, Dict[str, Any]] = {}
    for m in res.matches:
        meta = m.metadata if m.metadata is not None else {}
        if not isinstance(meta, dict):
            meta = dict(meta)

        context_text, child_text = _context_from_metadata(meta)
        parent_id = str(meta.get("parentId") or m.id)
        hit = {
            "text": context_text,
            "childText": child_text,
            "parentContext": context_text,
            "filename": meta.get("filename", ""),
            "parentId": parent_id,
            "score": m.score,
            "chunkStrategy": meta.get("chunkStrategy", ""),
        }

        existing = deduped.get(parent_id)
        if existing is None or (m.score or 0) > (existing.get("score") or 0):
            deduped[parent_id] = hit

    hits = sorted(deduped.values(), key=lambda x: x.get("score") or 0, reverse=True)
    return hits[:top_k]


def clear_pinecone_index() -> None:
    index, index_name = _pinecone_index()
    try:
        index.delete(delete_all=True)
        print(f"[jina→pinecone] cleared index {index_name!r}")
    except Exception as e:
        print(f"[jina→pinecone] clear warning: {e}")
