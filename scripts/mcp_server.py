#!/usr/bin/env python3
import os
import sys
import json
import platform
from typing import Optional

from mcp.server.fastmcp import FastMCP

# ==============================
#   إعداد MCP Server
# ==============================

mcp = FastMCP("FullFeatureMCP")

# مجلد آمن للملفات (عشان ما نخبص النظام)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "mcp_data"))
os.makedirs(BASE_DIR, exist_ok=True)


# ==============================
#   Resources (قراءة فقط)
# ==============================

@mcp.resource("system://health")
def system_health() -> dict:
    """حالة السيرفر الصحية."""
    return {
        "status": "ok",
        "message": "MCP server running",
    }


@mcp.resource("system://info")
def system_basic_info() -> dict:
    """معلومات بسيطة عن النظام."""
    return {
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "platform_release": platform.release(),
    }


@mcp.resource("utils://ping")
def ping() -> str:
    """Ping resource للتأكد أن السيرفر شغّال."""
    return "pong"


# ==============================
#   Tools — Utilities
# ==============================

@mcp.tool()
def sum_numbers(a: float, b: float) -> float:
    """جمع رقمين ويعيد الناتج."""
    return a + b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """ضرب رقمين ويعيد الناتج."""
    return a * b


@mcp.tool()
def echo(text: str) -> dict:
    """إرجاع النص كما هو داخل JSON."""
    return {"echo": text}


@mcp.tool()
def pretty_json(data: str) -> str:
    """
    تنسيق JSON string بشكل مرتب.
    - data: نص JSON (string) غير منسق.
    """
    try:
        obj = json.loads(data)
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception as e:
        return f"Invalid JSON: {e}"


# ==============================
#   Tools — HTTP Client بسيط
# ==============================

@mcp.tool()
def http_get(url: str, timeout: int = 10) -> dict:
    """
    تنفيذ طلب HTTP GET بسيط.
    - url: الرابط المطلوب.
    - timeout: المهلة بالثواني.
    """
    import requests

    try:
        resp = requests.get(url, timeout=timeout)
        content_type = resp.headers.get("Content-Type", "")
        # عشان ما نرجع body ضخم جدًا، نقصّه لو مرّة كبير
        text = resp.text
        max_len = 5000
        if len(text) > max_len:
            text = text[:max_len] + "\n...[truncated]..."
        return {
            "status_code": resp.status_code,
            "headers": dict(resp.headers),
            "content_type": content_type,
            "body": text,
        }
    except Exception as e:
        return {
            "error": str(e),
            "url": url,
        }


# ==============================
#   Tools — File Operations (داخل مجلد آمن)
# ==============================

def _safe_path(relative_path: str) -> str:
    """
    تحويل مسار نسبي إلى مسار آمن داخل BASE_DIR.
    يمنع الخروج من المجلد (no .. escape).
    """
    normalized = os.path.normpath(relative_path).lstrip(os.sep)
    full = os.path.abspath(os.path.join(BASE_DIR, normalized))
    if not full.startswith(BASE_DIR):
        raise ValueError("Invalid path (outside base dir)")
    return full


@mcp.tool()
def list_files(subdir: Optional[str] = "") -> list:
    """
    عرض الملفات داخل المجلد الآمن (mcp_data) أو مجلد فرعي.
    - subdir: مجلد فرعي اختياري.
    """
    path = _safe_path(subdir or "")
    if not os.path.exists(path):
        return []
    result = []
    for name in os.listdir(path):
        full = os.path.join(path, name)
        result.append({
            "name": name,
            "is_dir": os.path.isdir(full),
            "size": os.path.getsize(full) if os.path.isfile(full) else None,
        })
    return result


@mcp.tool()
def read_file(path: str, max_bytes: int = 5000) -> dict:
    """
    قراءة ملف نصي من داخل المجلد الآمن.
    - path: مسار نسبي من داخل mcp_data.
    - max_bytes: أقصى حجم يرجع.
    """
    full = _safe_path(path)
    if not os.path.exists(full):
        return {"error": "file_not_found", "path": path}
    if not os.path.isfile(full):
        return {"error": "not_a_file", "path": path}
    with open(full, "r", encoding="utf-8", errors="replace") as f:
        data = f.read(max_bytes + 1)
    truncated = False
    if len(data) > max_bytes:
        data = data[:max_bytes]
        truncated = True
    return {
        "path": path,
        "content": data,
        "truncated": truncated,
    }


@mcp.tool()
def write_file(path: str, content: str, overwrite: bool = True) -> dict:
    """
    كتابة محتوى إلى ملف داخل المجلد الآمن.
    - path: مسار نسبي من داخل mcp_data.
    - content: النص المراد كتابته.
    - overwrite: هل يسمح بالكتابة فوق ملف موجود.
    """
    full = _safe_path(path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    if os.path.exists(full) and not overwrite:
        return {"error": "file_exists", "path": path}
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)
    return {"status": "ok", "path": path}


# ==============================
#   ENTRYPOINT
# ==============================

def main():
    """
    أوامر التشغيل:
      python mcp_server.py          → stdio mode (افتراضي)
      python mcp_server.py stdio   → stdio mode
      python mcp_server.py http    → streamable-http mode
    """
    mode = "stdio"
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ("stdio", "http"):
            mode = "http" if arg == "http" else "stdio"

    if mode == "http":
        print("🔵 Starting MCP server with streamable-http …", file=sys.stderr)
        # ممكن تمرّر host/port عن طريق env لو حاب
        mcp.run(transport="streamable-http")
    else:
        print("🔵 Starting MCP server on stdio …", file=sys.stderr)
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
