import os
import re
import time
import uuid
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import httpx
from pinecone import Pinecone

load_dotenv()

class Chunk:
    def __init__(self, id: str, text: str, metadata: Dict[str, Any], embedding: Optional[List[float]] = None, parent_id: Optional[str] = None):
        self.id = id
        self.text = text
        self.metadata = metadata
        self.embedding = embedding
        self.parent_id = parent_id
        self.score = None

def cosine_similarity(v1, v2):
    if v1 is None or v2 is None:
        return 0
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    return dot_product / (norm_v1 * norm_v2)

class RAGService:
    def __init__(self, api_key: Optional[str] = None, pinecone_key: Optional[str] = None, index_name: Optional[str] = None):
        ak = (api_key or "").strip().strip("'").strip('"')
        if ak and ak not in ("undefined", "null"):
            import google.generativeai as genai
            genai.configure(api_key=ak)
        self.embedding_model = (os.getenv("GEMINI_EMBEDDING_MODEL") or "models/text-embedding-004").strip()
        self.jina_api_key = (os.getenv("JINA_API_KEY") or "").strip()
        self.jina_embeddings_url = (
            os.getenv("JINA_EMBEDDINGS_URL") or "https://api.jina.ai/v1/embeddings"
        ).strip()
        self.jina_embedding_model = (
            os.getenv("JINA_EMBEDDING_MODEL") or "jina-embeddings-v3"
        ).strip()
        try:
            self.jina_embed_batch = max(1, int(os.getenv("JINA_EMBED_BATCH") or "64"))
        except ValueError:
            self.jina_embed_batch = 64
        # Jina v3 支持 Matryoshka 式降维，与 Pinecone dimension=768 对齐（见 Jina Embeddings API 文档）
        try:
            self.jina_embedding_dimensions = int(
                (os.getenv("JINA_EMBEDDING_DIMENSIONS") or "768").strip()
            )
        except ValueError:
            self.jina_embedding_dimensions = 768
        if self.jina_embedding_dimensions < 1:
            self.jina_embedding_dimensions = 768
        try:
            self.semantic_merge_threshold = float(
                (os.getenv("SEMANTIC_CHUNK_SIM_THRESHOLD") or "0.75").strip()
            )
        except ValueError:
            self.semantic_merge_threshold = 0.75
        self.semantic_merge_threshold = max(0.5, min(0.99, self.semantic_merge_threshold))
        try:
            self.jina_http_timeout = float(
                (os.getenv("JINA_HTTP_TIMEOUT_SECONDS") or "300").strip()
            )
        except ValueError:
            self.jina_http_timeout = 300.0
        self.jina_http_timeout = max(30.0, self.jina_http_timeout)
        try:
            self.jina_embed_concurrency = max(
                1, int((os.getenv("JINA_EMBED_CONCURRENCY") or "4").strip())
            )
        except ValueError:
            self.jina_embed_concurrency = 4
        # Jina v3+ 必须指定 task，与 Milvus / Jina 官方说明一致：
        # https://milvus.io/docs/zh/embed-with-jina.md
        self.jina_task_passage = (
            os.getenv("JINA_TASK_PASSAGE") or "retrieval.passage"
        ).strip()
        self.jina_task_query = (os.getenv("JINA_TASK_QUERY") or "retrieval.query").strip()
        # 句级向量仅用于决定切点；默认与建库 passage 一致（同空间更稳），可用 JINA_TASK_SEMANTIC=text-matching 做句间匹配
        self.jina_task_semantic = (
            os.getenv("JINA_TASK_SEMANTIC") or "retrieval.passage"
        ).strip()
        self.index_name = index_name
        self.pc = None
        self.is_using_cloud = False
        self.local_vector_store: List[Chunk] = []

        if pinecone_key and self.index_name:
            try:
                self.pc = Pinecone(api_key=pinecone_key)
                self.is_using_cloud = True
                print(f"RAG: Connected to Pinecone index: {self.index_name}")
            except Exception as e:
                print(f"RAG: Failed to initialize Pinecone: {e}")
                raise RuntimeError(f"Pinecone 初始化失败: {e}") from e

        try:
            self.rag_chunk_size = max(100, int(os.getenv("RAG_CHUNK_SIZE") or "800"))
        except ValueError:
            self.rag_chunk_size = 800
        try:
            self.rag_chunk_overlap = max(0, int(os.getenv("RAG_CHUNK_OVERLAP") or "80"))
        except ValueError:
            self.rag_chunk_overlap = 80

    async def _jina_embed_texts(self, texts: List[str], task: str) -> List[List[float]]:
        if not self.jina_api_key:
            raise ValueError("JINA_API_KEY 未配置，无法调用 Jina 语义切割。")
        if not texts:
            return []

        bs = self.jina_embed_batch
        timeout = httpx.Timeout(self.jina_http_timeout, connect=60.0)
        sem = asyncio.Semaphore(self.jina_embed_concurrency)

        async def post_one(
            client: httpx.AsyncClient, batch_index: int, batch: List[str]
        ) -> tuple[int, List[List[float]]]:
            async with sem:
                body: Dict[str, Any] = {
                    "model": self.jina_embedding_model,
                    "input": batch,
                    "dimensions": self.jina_embedding_dimensions,
                    "task": task,
                }
                resp = await client.post(
                    self.jina_embeddings_url,
                    json=body,
                    headers={
                        "Authorization": f"Bearer {self.jina_api_key}",
                        "Content-Type": "application/json",
                    },
                )
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"Jina Embeddings API 错误 {resp.status_code}: {resp.text}"
                    )
                payload = resp.json()
                rows = payload.get("data") or []
                rows = sorted(rows, key=lambda x: x.get("index", 0))
                if len(rows) != len(batch):
                    raise RuntimeError(
                        f"Jina 返回 {len(rows)} 条向量，与输入条数 {len(batch)} 不一致"
                    )
                embs: List[List[float]] = []
                for row in rows:
                    emb = row["embedding"]
                    if len(emb) != self.jina_embedding_dimensions:
                        raise RuntimeError(
                            f"Jina 向量维度为 {len(emb)}，与配置的 JINA_EMBEDDING_DIMENSIONS={self.jina_embedding_dimensions} 不一致"
                        )
                    embs.append(emb)
                return batch_index, embs

        async with httpx.AsyncClient(timeout=timeout) as client:
            tasks = [
                post_one(client, i // bs, texts[i : i + bs])
                for i in range(0, len(texts), bs)
            ]
            parts = await asyncio.gather(*tasks)

        parts.sort(key=lambda x: x[0])
        out: List[List[float]] = []
        for _, embs in parts:
            out.extend(embs)
        return out

    async def get_embedding(self, text: str) -> List[float]:
        if not text or not text.strip():
            raise ValueError("Empty text for embedding")
        if self.jina_api_key:
            vecs = await self._jina_embed_texts(
                [text.strip()], self.jina_task_query
            )
            return vecs[0]
        import google.generativeai as genai
        result = genai.embed_content(
            model=self.embedding_model,
            content=text,
            task_type="retrieval_query",
            output_dimensionality=768
        )
        return result['embedding']

    async def parent_child_chunking(self, text: str, filename: str) -> List[Chunk]:
        if not text or not text.strip():
            return []

        parent_size = 1000
        child_size = 250
        chunks = []

        parent_texts = self._split_by_length(text, parent_size, 200)

        for i, parent_text in enumerate(parent_texts):
            parent_id = f"p-{int(time.time())}-{i}"
            child_texts = self._split_by_length(parent_text, child_size, 50)

            for j, child_text in enumerate(child_texts):
                if not child_text.strip():
                    continue
                chunks.append(Chunk(
                    id=f"c-{parent_id}-{j}",
                    text=child_text,
                    metadata={
                        "filename": filename,
                        "parentContext": parent_text,
                        "type": "child",
                        "text": child_text
                    },
                    parent_id=parent_id
                ))
        return chunks

    async def semantic_chunking(self, text: str, filename: str) -> List[Chunk]:
        if not text or not text.strip():
            return []

        sentences = re.split(r'(?<=[。！？\n])', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return []
        
        if len(sentences) < 2:
            return [Chunk(
                id=f"s-{int(time.time())}",
                text=text,
                metadata={"filename": filename, "text": text}
            )]

        try:
            if not self.jina_api_key:
                raise ValueError(
                    "semantic 策略必须配置 JINA_API_KEY；语义切割仅通过 Jina Embeddings API，不再使用 Gemini 做句子向量。"
                )
            embeddings = await self._jina_embed_texts(
                sentences, self.jina_task_semantic
            )

            chunks = []
            current_sentences = [sentences[0]]
            current_embedding = embeddings[0]
            threshold = self.semantic_merge_threshold

            for i in range(1, len(sentences)):
                sim = cosine_similarity(current_embedding, embeddings[i])
                if sim > threshold:
                    current_sentences.append(sentences[i])
                    # Update running avg embedding
                    current_embedding = [(a + b) / 2 for a, b in zip(current_embedding, embeddings[i])]
                else:
                    chunk_text = "".join(current_sentences)
                    chunks.append(Chunk(
                        id=f"sem-{int(time.time())}-{i}",
                        text=chunk_text,
                        metadata={"filename": filename, "text": chunk_text}
                    ))
                    current_sentences = [sentences[i]]
                    current_embedding = embeddings[i]

            if current_sentences:
                chunk_text = "".join(current_sentences)
                chunks.append(Chunk(
                    id=f"sem-final-{int(time.time())}",
                    text=chunk_text,
                    metadata={"filename": filename, "text": chunk_text}
                ))
            return chunks
        except Exception as e:
            print(f"Semantic chunking embedding failed: {e}")
            raise e

    def _split_by_length(self, text: str, size: int, overlap: int) -> List[str]:
        result = []
        start = 0
        while start < len(text):
            chunk = text[start : start + size]
            if chunk.strip():
                result.append(chunk)
            start += size - overlap
        return result

    def _text_chunks_for_index(self, text: str, filename: str) -> List[Chunk]:
        """将上传文本切成块，供 Jina → Pinecone 建库（与 test_jina_embeddings 单条逻辑一致，长文多块）。"""
        text = text.strip()
        if not text:
            return []

        parts = self._split_by_length(text, self.rag_chunk_size, self.rag_chunk_overlap)
        if not parts:
            parts = [text]

        run_id = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
        chunks: List[Chunk] = []
        for i, part in enumerate(parts):
            chunks.append(
                Chunk(
                    id=f"rag-{run_id}-{i}",
                    text=part,
                    metadata={
                        "text": part[:8000],
                        "filename": filename,
                        "source": "api/rag/index",
                        "model": self.jina_embedding_model,
                    },
                )
            )
        return chunks

    async def index_uploaded_text(self, text: str, filename: str) -> int:
        """
        与 test_jina_embeddings.py 相同链路：
        Jina Embeddings (retrieval.passage + dimensions) → Pinecone upsert。
        """
        if not self.jina_api_key:
            raise ValueError("JINA_API_KEY 未配置")
        if not self.is_using_cloud or not self.pc or not self.index_name:
            raise ValueError(
                "Pinecone 未就绪，请在 .env 配置 PINECONE_API_KEY 与 PINECONE_INDEX"
            )

        chunks = self._text_chunks_for_index(text, filename)
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        print(f"RAG: Jina embedding {len(texts)} chunk(s) for {filename!r}...")
        embeddings = await self._jina_embed_texts(texts, self.jina_task_passage)

        vectors = []
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb
            vectors.append(
                {
                    "id": chunk.id,
                    "values": emb,
                    "metadata": chunk.metadata,
                }
            )

        index = self.pc.Index(self.index_name)
        upserted = 0
        batch_size = 100
        for j in range(0, len(vectors), batch_size):
            batch_v = vectors[j : j + batch_size]
            res = index.upsert(vectors=batch_v)
            upserted += getattr(res, "upserted_count", None) or len(batch_v)

        self.local_vector_store.extend(chunks)
        print(f"RAG: Pinecone upserted {upserted} vector(s) into {self.index_name!r}")
        return upserted

    async def search(self, query: str, limit: int = 3) -> List[Chunk]:
        query_embedding = await self.get_embedding(query)

        if self.is_using_cloud and self.pc:
            index = self.pc.Index(self.index_name)
            res = index.query(
                vector=query_embedding,
                top_k=limit,
                include_metadata=True
            )
            results = []
            for m in res.matches:
                meta = m.metadata if m.metadata is not None else {}
                if not isinstance(meta, dict):
                    meta = dict(meta)
                results.append(Chunk(
                    id=m.id,
                    text=meta.get("text", ""),
                    metadata=meta,
                    embedding=None,
                ))
                results[-1].score = m.score
            return results
        else:
            if not self.local_vector_store:
                return []
            
            scored = []
            for c in self.local_vector_store:
                score = cosine_similarity(query_embedding, c.embedding)
                c_copy = Chunk(c.id, c.text, c.metadata, None, c.parent_id)
                c_copy.score = score
                scored.append(c_copy)
            
            scored.sort(key=lambda x: x.score, reverse=True)
            return scored[:limit]

    def clear(self):
        if self.is_using_cloud and self.pc:
            index = self.pc.Index(self.index_name)
            try:
                index.delete(delete_all=True)
            except:
                # Some index types don't support delete_all
                pass
        self.local_vector_store = []
