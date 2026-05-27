"""Quick DashScope / Qwen smoke test (same payload shape as main.py)."""
import json
import os
import sys

import httpx
from dotenv import load_dotenv

load_dotenv()


def clean_key(key: str | None) -> str:
    if not key or key in ("undefined", "null"):
        return ""
    return key.strip().strip("'").strip('"')


def main() -> int:
    key = clean_key(os.getenv("DASHSCOPE_API_KEY"))
    from main import QWEN_MODEL, dashscope_url_for_model, format_dashscope_messages, uses_multimodal_endpoint

    model = QWEN_MODEL
    url = dashscope_url_for_model(model)
    multimodal = uses_multimodal_endpoint(model)

    if not key:
        print("错误: 未设置 DASHSCOPE_API_KEY", file=sys.stderr)
        return 1

    print(f"URL: {url}")
    print(f"model: {model} (multimodal={multimodal})")
    print(f"key: {key[:12]}...")

    messages = [
        {"role": "system", "content": "你是数据库助教"},
        {"role": "user", "content": "你好，请用一句话介绍你自己。"},
    ]
    messages = format_dashscope_messages(messages, multimodal)
    payload = {
        "model": model,
        "input": {"messages": messages},
        "parameters": {"result_format": "message", "incremental_output": False},
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    try:
        resp = httpx.post(url, headers=headers, json=payload, timeout=120.0)
    except httpx.RequestError as e:
        print(f"网络错误: {e}", file=sys.stderr)
        return 2

    print(f"HTTP {resp.status_code}")
    try:
        data = resp.json()
        print(json.dumps(data, ensure_ascii=False, indent=2)[:3000])
    except Exception:
        print(resp.text[:3000])

    if resp.status_code != 200:
        return 3

    content = (
        data.get("output", {})
        .get("choices", [{}])[0]
        .get("message", {})
        .get("content")
    )
    if content:
        print("\n--- 模型回复 ---")
        print(content)
        print("\nQwen API OK")
        return 0

    print("\n200 但无法解析 output.choices[0].message.content", file=sys.stderr)
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
