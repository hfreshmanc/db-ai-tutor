"""从上传文件中提取纯文本，供 RAG 建库使用。"""
from io import BytesIO
from typing import Iterable

TEXT_EXTENSIONS = {".txt", ".md", ".sql", ".csv", ".json"}
PDF_EXTENSIONS = {".pdf"}


def _ext(filename: str) -> str:
    dot = filename.rfind(".")
    if dot == -1:
        return ""
    return filename[dot:].lower()


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "gbk", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("无法解码文本文件，请使用 UTF-8 编码。")


def _extract_pdf_text(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ValueError("服务端未安装 pypdf，请执行 pip install pypdf") from e

    reader = PdfReader(BytesIO(data))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as e:
            raise ValueError("PDF 已加密，请先解密后再上传。") from e

    parts: list[str] = []
    for page in reader.pages:
        page_text = (page.extract_text() or "").strip()
        if page_text:
            parts.append(page_text)

    if not parts:
        raise ValueError(
            "未能从 PDF 提取文本。可能是扫描版图片 PDF，当前仅支持可选中文字的 PDF。"
        )
    return "\n\n".join(parts)


def extract_document_text(filename: str, data: bytes) -> str:
    if not data:
        raise ValueError("文件内容为空。")

    ext = _ext(filename or "")
    if ext in PDF_EXTENSIONS:
        text = _extract_pdf_text(data)
    elif ext in TEXT_EXTENSIONS or not ext:
        text = _decode_text(data)
    else:
        supported = ", ".join(sorted(TEXT_EXTENSIONS | PDF_EXTENSIONS))
        raise ValueError(f"不支持的文件类型 {ext or '(无扩展名)'}，当前支持：{supported}")

    text = text.strip()
    if not text:
        raise ValueError("提取到的文本为空，请检查文件内容。")
    return text


def supported_upload_extensions() -> Iterable[str]:
    return sorted(TEXT_EXTENSIONS | PDF_EXTENSIONS)
