"""
文件解析模块：上传文件 → 结构化内容

支持类型：
  - 文本类：txt / md / csv / json / log / py / html / xml / yaml / ini / sql ...
  - 文档类：pdf（pypdf）、docx / pptx（zipfile 提取 XML 文本）、xlsx（openpyxl）
  - 图片类：png / jpg / jpeg / gif / webp / bmp（转为 base64 data URL，交给视觉模型）
  - 压缩包：zip（列出包含的文件清单）
  - 视频 / 音频：上传成功但暂不支持内容解析（返回说明文字，AI 可基于文件名/元信息作答）
  - 旧版 Office（doc / xls / ppt）：提示另存为新版格式

返回统一结构：
  {
    "filename": str,
    "kind": "text" | "image",
    "content": str,        # 文本内容 或 data URL
    "mime": str,
    "size": int,
    "note": str,           # 说明/警告
  }
"""
import base64
import io
import json
import logging
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# 大小上限（按类型）
MAX_IMAGE_SIZE = 10 * 1024 * 1024      # 图片 10MB（需转 base64）
MAX_DOC_SIZE = 50 * 1024 * 1024        # 文档/文本 50MB
MAX_MEDIA_SIZE = 200 * 1024 * 1024     # 视频/音频 200MB
# 提取文本后截断长度（防止超长 token）
MAX_TEXT_CHARS = 60000

TEXT_EXTS = {".txt", ".md", ".markdown", ".csv", ".json", ".log", ".py", ".js",
             ".html", ".htm", ".xml", ".yaml", ".yml", ".ini", ".cfg", ".conf",
             ".tsv", ".sql", ".sh", ".bat", ".css", ".rtf", ".tex"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
IMAGE_MIME = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
}
DOC_EXTS = {".pdf", ".docx", ".xlsx", ".pptx"}
LEGACY_OFFICE_EXTS = {".doc", ".xls", ".ppt"}
VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm",
              ".m4v", ".ts", ".mpg", ".mpeg", ".3gp"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".wma", ".amr"}
ARCHIVE_EXTS = {".zip"}

VIDEO_MIME = {
    ".mp4": "video/mp4", ".avi": "video/x-msvideo", ".mkv": "video/x-matroska",
    ".mov": "video/quicktime", ".wmv": "video/x-ms-wmv", ".flv": "video/x-flv",
    ".webm": "video/webm", ".m4v": "video/x-m4v", ".ts": "video/mp2t",
    ".mpg": "video/mpeg", ".mpeg": "video/mpeg", ".3gp": "video/3gpp",
}
AUDIO_MIME = {
    ".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4",
    ".aac": "audio/aac", ".ogg": "audio/ogg", ".flac": "audio/flac",
    ".wma": "audio/x-ms-wma", ".amr": "audio/amr",
}


def parse_file(filename: str, data: bytes) -> Dict[str, object]:
    """解析上传文件内容。"""
    name = Path(filename).name
    ext = Path(name).suffix.lower()
    size = len(data)

    # 图片：转 base64 data URL
    if ext in IMAGE_EXTS:
        if size > MAX_IMAGE_SIZE:
            return _meta_result(name, size, "image", f"图片超过 {MAX_IMAGE_SIZE // 1024 // 1024}MB 限制，未解析")
        mime = IMAGE_MIME.get(ext, "image/png")
        b64 = base64.b64encode(data).decode("ascii")
        data_url = f"data:{mime};base64,{b64}"
        return {"filename": name, "kind": "image", "content": data_url,
                "mime": mime, "size": size, "note": ""}

    # 视频 / 音频：暂不支持内容解析
    if ext in VIDEO_EXTS or ext in AUDIO_EXTS:
        kind_name = "视频" if ext in VIDEO_EXTS else "音频"
        limit = MAX_MEDIA_SIZE // 1024 // 1024
        if size > MAX_MEDIA_SIZE:
            return _meta_result(name, size, kind_name, f"文件超过 {limit}MB 限制，未接收")
        mime = (VIDEO_MIME if ext in VIDEO_EXTS else AUDIO_MIME).get(ext, "application/octet-stream")
        hint = (f"[{kind_name}文件] 文件名：{name}，大小：{_fmt_size(size)}。"
                f"当前版本暂不支持解析{kind_name}内容，建议补充文字说明，"
                f"或将关键画面截图后上传，我同样可以帮你分析。")
        return {"filename": name, "kind": "text", "content": hint,
                "mime": mime, "size": size,
                "note": f"{kind_name}内容暂不支持解析（已记录文件名）"}

    # 压缩包：列出文件清单
    if ext in ARCHIVE_EXTS:
        if size > MAX_DOC_SIZE:
            return _meta_result(name, size, "压缩包", f"文件超过 {MAX_DOC_SIZE // 1024 // 1024}MB 限制，未解析")
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                names = z.namelist()
            listing = "\n".join(names[:200])
            more = f"\n……共 {len(names)} 个条目" if len(names) > 200 else ""
            content = (f"[压缩包] {name}\n包含 {len(names)} 个文件/文件夹：\n{listing}{more}\n"
                       f"（如需分析其中某个文件，请解压后单独上传）")
            return {"filename": name, "kind": "text", "content": content,
                    "mime": "application/zip", "size": size, "note": ""}
        except Exception as ex:
            return _meta_result(name, size, "压缩包", f"压缩包读取失败：{ex}")

    # 旧版 Office：提示转存
    if ext in LEGACY_OFFICE_EXTS:
        new_ext = {".doc": ".docx", ".xls": ".xlsx", ".ppt": ".pptx"}[ext]
        return _meta_result(
            name, size, "旧版Office",
            f"旧版 {ext.upper()} 格式暂不支持直接解析，请在 Office/WPS 中另存为 {new_ext} 后重新上传",
        )

    # 文本类
    if ext in TEXT_EXTS:
        text = _decode_text(data)
        return {"filename": name, "kind": "text", "content": text,
                "mime": "text/plain", "size": size, "note": ""}

    # 文档类
    if ext in DOC_EXTS:
        if size > MAX_DOC_SIZE:
            return _meta_result(name, size, "文档", f"文件超过 {MAX_DOC_SIZE // 1024 // 1024}MB 限制，未解析")
        try:
            if ext == ".pdf":
                text = _parse_pdf(data)
                mime = "application/pdf"
            elif ext == ".docx":
                text = _parse_docx(data)
                mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            elif ext == ".xlsx":
                text = _parse_xlsx(data)
                mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            elif ext == ".pptx":
                text = _parse_pptx(data)
                mime = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
            else:
                text, mime = "", "application/octet-stream"

            if not text.strip():
                return {"filename": name, "kind": "text", "content": "",
                        "mime": mime, "size": size,
                        "note": f"{ext.upper()} 文件未能提取到文本（可能是扫描件或加密文件）"}
            return {"filename": name, "kind": "text", "content": text,
                    "mime": mime, "size": size, "note": ""}
        except Exception as ex:
            logger.warning(f"解析 {name} 失败: {ex}")
            return _meta_result(name, size, "文档", f"解析失败：{ex}")

    # 其他：尝试按文本读取
    try:
        text = _decode_text(data)
        return {"filename": name, "kind": "text", "content": text,
                "mime": "application/octet-stream", "size": size,
                "note": f"未知文件类型（{ext or '无扩展名'}），已按文本读取"}
    except Exception:
        return _meta_result(name, size, "文件", f"不支持的文件类型（{ext or '无扩展名'}）")


def _meta_result(filename: str, size: int, category: str, note: str) -> Dict[str, object]:
    """构造一个未解析成功但可用的元信息结果。"""
    return {
        "filename": filename, "kind": "text",
        "content": f"[{category}文件] 文件名：{filename}，大小：{_fmt_size(size)}。{note}",
        "mime": "application/octet-stream", "size": size, "note": note,
    }


def _fmt_size(size: int) -> str:
    if size >= 1024 * 1024:
        return f"{size / 1024 / 1024:.1f}MB"
    if size >= 1024:
        return f"{size / 1024:.1f}KB"
    return f"{size}B"


def _decode_text(data: bytes) -> str:
    """按顺序尝试 UTF-8 / GBK / latin-1 解码，并截断。"""
    for enc in ("utf-8", "gbk", "latin-1"):
        try:
            text = data.decode(enc)
            break
        except (UnicodeDecodeError, LookupError):
            continue
    else:
        text = data.decode("utf-8", errors="replace")
    return text[:MAX_TEXT_CHARS]


def _parse_pdf(data: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
            pages.append(t)
        except Exception:
            continue
    text = "\n\n".join(pages)
    return text[:MAX_TEXT_CHARS]


def _parse_docx(data: bytes) -> str:
    """DOCX 本质是 zip，提取 word/document.xml 中的文本。"""
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        xml = z.read("word/document.xml").decode("utf-8", errors="replace")
    # 段落换行
    xml = re.sub(r"</w:p>", "\n", xml)
    xml = re.sub(r"<w:tab[^>]*/>", "\t", xml)
    text = re.sub(r"<[^>]+>", "", xml)
    return _clean_ws(text)


def _parse_xlsx(data: bytes) -> str:
    from openpyxl import load_workbook
    wb = load_workbook(io.BytesIO(data), data_only=True, read_only=True)
    rows_out = []
    for ws in wb.worksheets:
        rows_out.append(f"【工作表：{ws.title}】")
        for row in ws.iter_rows(values_only=True):
            cells = ["" if v is None else str(v) for v in row]
            if any(c.strip() for c in cells):
                rows_out.append(" | ".join(cells))
    return "\n".join(rows_out)[:MAX_TEXT_CHARS]


def _parse_pptx(data: bytes) -> str:
    """PPTX 本质是 zip，按幻灯片顺序提取文本。"""
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        slide_files = sorted(
            [n for n in z.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", n)],
            key=lambda n: int(re.search(r"(\d+)", n).group(1)),
        )
        parts = []
        for sf in slide_files:
            xml = z.read(sf).decode("utf-8", errors="replace")
            xml = re.sub(r"</a:p>", "\n", xml)
            text = re.sub(r"<[^>]+>", "", xml)
            if text.strip():
                parts.append(text.strip())
    return "\n\n".join(parts)[:MAX_TEXT_CHARS]


def _clean_ws(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()[:MAX_TEXT_CHARS]
