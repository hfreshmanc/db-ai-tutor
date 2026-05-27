"""
最小脚本：验证 Jina Embeddings → Pinecone upsert 全链路。
用法（在项目根目录）:
  pip install requests python-dotenv pinecone
  python test_jina_embeddings.py

环境变量（.env）:
  JINA_API_KEY
  PINECONE_API_KEY
  PINECONE_INDEX
  可选: JINA_EMBEDDING_MODEL, JINA_EMBEDDINGS_URL, JINA_EMBEDDING_DIMENSIONS
"""
import os
import sys
import time
import uuid
from typing import Optional

from dotenv import load_dotenv
import requests
from pinecone import Pinecone

load_dotenv()

JINA_API_KEY = (os.getenv("JINA_API_KEY") or "").strip()
PINECONE_API_KEY = (os.getenv("PINECONE_API_KEY") or "").strip()
PINECONE_INDEX = (os.getenv("PINECONE_INDEX") or "").strip()

URL = (os.getenv("JINA_EMBEDDINGS_URL") or "https://api.jina.ai/v1/embeddings").strip()
MODEL = (os.getenv("JINA_EMBEDDING_MODEL") or "jina-embeddings-v3").strip()
TEST_TEXT = "你好，这是一个RAG测试"

if not JINA_API_KEY:
    print("错误: 未设置 JINA_API_KEY。", file=sys.stderr)
    sys.exit(1)
if not PINECONE_API_KEY or not PINECONE_INDEX:
    print("错误: 请配置 PINECONE_API_KEY 与 PINECONE_INDEX。", file=sys.stderr)
    sys.exit(1)

dim_raw = (os.getenv("JINA_EMBEDDING_DIMENSIONS") or "").strip()
dimensions: Optional[int] = None
if dim_raw:
    try:
        dimensions = int(dim_raw)
    except ValueError:
        print(f"警告: JINA_EMBEDDING_DIMENSIONS={dim_raw!r} 无效，将使用 Jina 默认维度。", file=sys.stderr)

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {JINA_API_KEY}",
}

body: dict = {
    "model": MODEL,
    "input": [TEST_TEXT],
    "task": "retrieval.passage",
}
if dimensions is not None:
    body["dimensions"] = dimensions

print("1/2 请求 Jina Embeddings...")
try:
    response = requests.post(URL, headers=headers, json=body, timeout=120)
except requests.RequestException as e:
    print(f"网络错误: {e}", file=sys.stderr)
    sys.exit(2)

if response.status_code != 200:
    print(f"Jina HTTP {response.status_code}: {response.text}", file=sys.stderr)
    sys.exit(3)

result = response.json()
rows = result.get("data") or []
if not rows or "embedding" not in rows[0]:
    print(f"Jina 响应格式异常: {result}", file=sys.stderr)
    sys.exit(4)

embedding = rows[0]["embedding"]
print("  Jina 成功")
print(f"  模型: {MODEL}")
print(f"  向量维度: {len(embedding)}")
print(f"  前 5 维: {embedding[:5]}")

vector_id = f"test-jina-{int(time.time())}-{uuid.uuid4().hex[:8]}"
metadata = {
    "text": TEST_TEXT,
    "source": "test_jina_embeddings.py",
    "model": MODEL,
}

print(f"2/2 写入 Pinecone 索引 {PINECONE_INDEX!r} (id={vector_id})...")
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)
    upsert_res = index.upsert(
        vectors=[
            {
                "id": vector_id,
                "values": embedding,
                "metadata": metadata,
            }
        ]
    )
except Exception as e:
    print(f"Pinecone 错误: {e}", file=sys.stderr)
    sys.exit(5)

count = getattr(upsert_res, "upserted_count", None)
print("  Pinecone upsert 成功")
if count is not None:
    print(f"  upserted_count: {count}")

print("\n全链路 OK：Jina → Pinecone")
