import os
from dotenv import load_dotenv

load_dotenv()

import asyncio
import json
import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from jina_pinecone_index import (
    index_text_to_pinecone,
    search_pinecone,
    clear_pinecone_index,
    get_chunk_strategy,
)
from document_parser import extract_document_text, supported_upload_extensions
from prompts import (
    get_base_system_prompt,
    get_rag_system_prompt,
    RAG_USER_NO_HITS,
    RAG_USER_WITH_CONTEXT,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    """用于确认当前跑的是 Jina 版后端，而非旧 Gemini 进程。"""
    return {
        "ok": True,
        "pipeline": "jina→pinecone",
        "chunking": get_chunk_strategy(),
        "chat": "dashscope-qwen",
        "qwen_model": QWEN_MODEL,
        "dashscope_endpoint": "multimodal"
        if uses_multimodal_endpoint(QWEN_MODEL)
        else "text",
        "pinecone_index": PINECONE_INDEX or None,
        "upload_formats": list(supported_upload_extensions()),
    }


def clean_key(key: Optional[str]) -> str:
    if not key or key in ["undefined", "null"]:
        return ""
    return key.strip().strip("'").strip('"')


PINECONE_KEY = clean_key(os.getenv("PINECONE_API_KEY"))
PINECONE_INDEX = clean_key(os.getenv("PINECONE_INDEX"))
DEFAULT_DASHSCOPE_KEY = clean_key(os.getenv("DASHSCOPE_API_KEY"))

DASHSCOPE_TEXT_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
)
DASHSCOPE_MULTIMODAL_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
)
QWEN_MODEL = (os.getenv("QWEN_MODEL") or "qwen3.6-plus").strip()

try:
    RAG_TOP_K = max(1, int((os.getenv("RAG_TOP_K") or "3").strip()))
except ValueError:
    RAG_TOP_K = 3


def resolve_system_instruction(client_instruction: Optional[str] = None) -> str:
    """统一使用服务端 Prompt；忽略前端传入的简短占位文案。"""
    _ = client_instruction
    return get_base_system_prompt()


def format_rag_context(hits: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for i, h in enumerate(hits, 1):
        fname = h.get("filename") or "未知文件"
        context = (h.get("parentContext") or h.get("text") or "").strip()
        child = (h.get("childText") or "").strip()
        score = h.get("score")
        score_hint = f" | 相关度 {score:.3f}" if isinstance(score, (int, float)) else ""
        block = f"--- 片段 {i} | 来源: {fname}{score_hint} ---\n{context}"
        if child and context and child != context:
            preview = child if len(child) <= 300 else child[:300] + "..."
            block += f"\n（命中子片段：{preview}）"
        lines.append(block)
    return "\n\n".join(lines)


def build_rag_prompt(
    user_message: str,
    hits: List[Dict[str, Any]],
    base_system: str,
) -> tuple[str, str]:
    """RAG 开启：system 追加 RAG 规则；user 注入参考资料与用户问题。"""
    _ = base_system
    system = get_rag_system_prompt()
    if hits:
        context_str = format_rag_context(hits)
        user = RAG_USER_WITH_CONTEXT.format(
            context=context_str,
            question=user_message.strip(),
        )
    else:
        user = RAG_USER_NO_HITS.format(question=user_message.strip())
    return system, user


def uses_multimodal_endpoint(model: str) -> bool:
    """qwen3.6-plus 等需 multimodal-generation，纯文本 qwen-plus 用 text-generation。"""
    m = model.lower()
    if "vl" in m or "-ocr" in m:
        return True
    if m.startswith("qwen3.6") or m.startswith("qwen3.5"):
        return True
    return False


def dashscope_url_for_model(model: str) -> str:
    explicit = (os.getenv("DASHSCOPE_API_URL") or "").strip()
    if explicit:
        return explicit
    if uses_multimodal_endpoint(model):
        return DASHSCOPE_MULTIMODAL_URL
    return DASHSCOPE_TEXT_URL


def format_dashscope_messages(
    messages: List[Dict[str, Any]], multimodal: bool
) -> List[Dict[str, Any]]:
    if not multimodal:
        return messages
    formatted: List[Dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            formatted.append({"role": msg["role"], "content": [{"text": content}]})
        else:
            formatted.append(msg)
    return formatted


def normalize_dashscope_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """多模态 API 的 content 为 [{text: ...}]，归一化为字符串供前端解析。"""
    try:
        message = data["output"]["choices"][0]["message"]
        content = message.get("content")
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("text"):
                    parts.append(str(item["text"]))
                elif isinstance(item, str):
                    parts.append(item)
            message["content"] = "".join(parts)
    except (KeyError, IndexError, TypeError):
        pass
    return data


class IndexRequest(BaseModel):
    text: str
    filename: str


def _ensure_rag_ready() -> None:
    if not (os.getenv("JINA_API_KEY") or "").strip():
        raise HTTPException(status_code=400, detail="请配置 JINA_API_KEY。")
    if not PINECONE_KEY or not PINECONE_INDEX:
        raise HTTPException(
            status_code=400,
            detail="请配置 PINECONE_API_KEY 与 PINECONE_INDEX。",
        )


async def _index_text_response(text: str, filename: str) -> Dict[str, Any]:
    count = await asyncio.to_thread(index_text_to_pinecone, text, filename)
    return {
        "success": True,
        "count": count,
        "filename": filename,
        "textLength": len(text),
        "pinecone_index": PINECONE_INDEX,
        "pipeline": "jina→pinecone",
        "chunking": get_chunk_strategy(),
    }


@app.post("/api/rag/index")
async def rag_index(req_data: IndexRequest):
    _ensure_rag_ready()
    if not req_data.text or not req_data.text.strip():
        raise HTTPException(status_code=400, detail="上传内容为空。")

    try:
        return await _index_text_response(req_data.text, req_data.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rag/index/upload")
async def rag_index_upload(file: UploadFile = File(...)):
    """multipart 上传：支持 .txt/.md/.sql/.pdf 等，PDF 在后端解析为文本后建库。"""
    _ensure_rag_ready()

    filename = (file.filename or "upload").strip()
    if not filename:
        raise HTTPException(status_code=400, detail="文件名为空。")

    try:
        raw = await file.read()
        text = extract_document_text(filename, raw)
        resp = await _index_text_response(text, filename)
        resp["sourceType"] = "pdf" if filename.lower().endswith(".pdf") else "text"
        return resp
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rag/clear")
async def rag_clear():
    if not PINECONE_KEY or not PINECONE_INDEX:
        raise HTTPException(status_code=400, detail="请配置 Pinecone。")
    try:
        await asyncio.to_thread(clear_pinecone_index)
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, Any]]
    systemInstruction: Optional[str] = None
    stream: bool = False
    useRAG: bool = False
    dashscopeKey: Optional[str] = None


@app.post("/api/chat")
async def chat(req_data: ChatRequest):
    dashscope_key = clean_key(req_data.dashscopeKey) or DEFAULT_DASHSCOPE_KEY
    if not dashscope_key:
        raise HTTPException(
            status_code=401,
            detail="请配置 DASHSCOPE_API_KEY 或在设置中填写通义千问 Key。",
        )

    system_instruction = resolve_system_instruction(req_data.systemInstruction)
    user_message = req_data.message

    if req_data.useRAG:
        if not (os.getenv("JINA_API_KEY") or "").strip():
            raise HTTPException(status_code=400, detail="RAG 需要 JINA_API_KEY。")
        if not PINECONE_KEY or not PINECONE_INDEX:
            raise HTTPException(status_code=400, detail="RAG 需要 Pinecone 配置。")

        # 1) 用户问题 → Jina(query) 向量化 → Pinecone Top-K 检索
        hits = await asyncio.to_thread(search_pinecone, req_data.message, RAG_TOP_K)
        # 2) 检索片段写入 system/user prompt，约束模型优先依据资料作答
        system_instruction, user_message = build_rag_prompt(
            req_data.message, hits, system_instruction
        )

    return await handle_qwen_chat(
        user_message,
        req_data.history,
        system_instruction,
        dashscope_key,
        req_data.stream,
    )


async def handle_qwen_chat(
    message: str, history: List[Dict], system: str, key: str, stream: bool
):
    """DashScope：纯文本走 text-generation；qwen3.6-plus 等走 multimodal-generation。"""
    multimodal = uses_multimodal_endpoint(QWEN_MODEL)
    url = dashscope_url_for_model(QWEN_MODEL)
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "X-DashScope-SSE": "enable" if stream else "disable",
    }

    messages: List[Dict[str, Any]] = [{"role": "system", "content": system}]
    for h in history:
        messages.append({"role": h["role"], "content": h["parts"][0]["text"]})
    messages.append({"role": "user", "content": message})
    messages = format_dashscope_messages(messages, multimodal)

    payload: Dict[str, Any] = {
        "model": QWEN_MODEL,
        "input": {"messages": messages},
        "parameters": {
            "result_format": "message",
            "incremental_output": bool(stream),
        },
    }

    if not stream:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, headers=headers, json=payload, timeout=120.0)
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            return normalize_dashscope_response(resp.json())

    async def generate():
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST", url, headers=headers, json=payload, timeout=120.0
            ) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data:"):
                        yield line + "\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


if os.path.exists("dist"):
    app.mount("/", StaticFiles(directory="dist", html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=3001)
