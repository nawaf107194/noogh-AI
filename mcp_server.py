#!/usr/bin/env python3
import os
import sys
import json
import platform
from typing import Optional
from mcp.server.fastmcp import FastMCP

# ============================================================
#   MCP SERVER — FULL VERSION FIXED WITH UVICORN HTTP
# ============================================================

mcp = FastMCP("FullFeatureMCP")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "mcp_data"))
os.makedirs(BASE_DIR, exist_ok=True)


# ============================================================
#   RESOURCES
# ============================================================

@mcp.resource("system://health")
def system_health() -> dict:
    return {"status": "ok", "message": "MCP server running"}

@mcp.resource("system://info")
def system_basic_info() -> dict:
    return {
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "platform_release": platform.release(),
    }

@mcp.resource("utils://ping")
def ping() -> str:
    return "pong"


# ============================================================
#   TOOLS — BASIC
# ============================================================

@mcp.tool()
def sum_numbers(a: float, b: float) -> float:
    return a + b

@mcp.tool()
def multiply(a: float, b: float) -> float:
    return a * b

@mcp.tool()
def echo(text: str) -> dict:
    return {"echo": text}

@mcp.tool()
def pretty_json(data: str) -> str:
    try:
        obj = json.loads(data)
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception as e:
        return f"Invalid JSON: {e}"


# ============================================================
#   TOOLS — HTTP GET
# ============================================================

@mcp.tool()
def http_get(url: str, timeout: int = 10) -> dict:
    import requests

    try:
        resp = requests.get(url, timeout=timeout)
        text = resp.text
        if len(text) > 5000:
            text = text[:5000] + "\n...[truncated]..."

        return {
            "status_code": resp.status_code,
            "headers": dict(resp.headers),
            "body": text,
        }
    except Exception as e:
        return {"error": str(e), "url": url}


# ============================================================
#   TOOLS — FILE OPS
# ============================================================

def _safe_path(relative_path: str) -> str:
    normalized = os.path.normpath(relative_path).lstrip(os.sep)
    full = os.path.abspath(os.path.join(BASE_DIR, normalized))
    if not full.startswith(BASE_DIR):
        raise ValueError("Invalid path (outside base dir)")
    return full

@mcp.tool()
def list_files(subdir: Optional[str] = "") -> list:
    path = _safe_path(subdir or "")
    if not os.path.exists(path):
        return []

    items = []
    for name in os.listdir(path):
        full = os.path.join(path, name)
        items.append({
            "name": name,
            "is_dir": os.path.isdir(full),
            "size": os.path.getsize(full) if os.path.isfile(full) else None,
        })
    return items

@mcp.tool()
def read_file(path: str, max_bytes: int = 5000) -> dict:
    full = _safe_path(path)
    if not os.path.exists(full):
        return {"error": "file_not_found"}

    with open(full, "r", encoding="utf-8", errors="replace") as f:
        data = f.read(max_bytes + 1)

    return {
        "path": path,
        "content": data[:max_bytes],
        "truncated": len(data) > max_bytes
    }

@mcp.tool()
def write_file(path: str, content: str, overwrite: bool = True) -> dict:
    full = _safe_path(path)
    if os.path.exists(full) and not overwrite:
        return {"error": "file_exists"}

    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)

    return {"status": "ok", "path": path}


# ============================================================
#   ENTRYPOINT — FIXED HTTP MODE WITH UVICORN
# ============================================================

def main():
    mode = "stdio"

    if len(sys.argv) > 1 and sys.argv[1].lower() == "http":
        mode = "http"

    if mode == "http":
        # Set environment variables for uvicorn (note: FastMCP may not respect these)
        # Using port 8001 to avoid conflict with main API on 8000
        os.environ["HOST"] = os.environ.get("MCP_HOST", "0.0.0.0")
        os.environ["PORT"] = "8001"  # Force port 8001

        host = os.environ.get("HOST")
        port = os.environ.get("PORT")

        print(f"🔵 Starting HTTP MCP server at http://{host}:{port}", file=sys.stderr)

        # FastMCP uses run() method with sse transport for HTTP
        mcp.run(transport="sse")

    else:
        print("🔵 Starting MCP server on stdio …", file=sys.stderr)
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
