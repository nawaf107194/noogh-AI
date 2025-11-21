#!/usr/bin/env python3
"""
= MCP Server v2.0 - Enhanced Production Version

FastMCP-based server with full health monitoring and tool suite
"""

import os
import sys
import json
import platform
import logging
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print("L Error: mcp package not found. Install: pip install mcp", file=sys.stderr)
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [MCP] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler('logs/mcp.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# 
# Configuration
# 

mcp = FastMCP("NooghUnifiedMCP")

BASE_DIR = Path(__file__).parent.parent / "mcp_data"
BASE_DIR.mkdir(exist_ok=True)

# Track server stats
SERVER_STATS = {
    "start_time": datetime.now().isoformat(),
    "requests_total": 0,
    "requests_success": 0,
    "requests_failed": 0,
}

# 
# Resources (Read-only endpoints)
# 

@mcp.resource("system://health")
def system_health() -> Dict[str, Any]:
    """Health check endpoint"""
    uptime = (datetime.now() - datetime.fromisoformat(SERVER_STATS["start_time"])).total_seconds()
    return {
        "status": "healthy",
        "uptime_seconds": uptime,
        "version": "2.0",
        "timestamp": datetime.now().isoformat()
    }

@mcp.resource("system://info")
def system_info() -> Dict[str, Any]:
    """System information"""
    return {
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "platform_release": platform.release(),
        "server": "NooghUnifiedMCP",
        "version": "2.0"
    }

@mcp.resource("system://stats")
def system_stats() -> Dict[str, Any]:
    """Server statistics"""
    return SERVER_STATS.copy()

@mcp.resource("utils://ping")
def ping() -> str:
    """Simple ping endpoint"""
    return "pong"

# 
# Tools - Math & Utilities
# 

@mcp.tool()
def sum_numbers(a: float, b: float) -> float:
    """Add two numbers"""
    SERVER_STATS["requests_total"] += 1
    SERVER_STATS["requests_success"] += 1
    return a + b

@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers"""
    SERVER_STATS["requests_total"] += 1
    SERVER_STATS["requests_success"] += 1
    return a * b

@mcp.tool()
def echo(text: str) -> Dict[str, str]:
    """Echo back the input text"""
    SERVER_STATS["requests_total"] += 1
    SERVER_STATS["requests_success"] += 1
    return {"echo": text, "timestamp": datetime.now().isoformat()}

@mcp.tool()
def pretty_json(data: str) -> str:
    """Format JSON with proper indentation"""
    SERVER_STATS["requests_total"] += 1
    try:
        obj = json.loads(data)
        SERVER_STATS["requests_success"] += 1
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception as e:
        SERVER_STATS["requests_failed"] += 1
        return f"Invalid JSON: {e}"

# 
# Tools - HTTP Operations
# 

@mcp.tool()
def http_get(url: str, timeout: int = 10) -> Dict[str, Any]:
    """Make HTTP GET request"""
    import requests

    SERVER_STATS["requests_total"] += 1
    try:
        resp = requests.get(url, timeout=timeout)
        text = resp.text
        if len(text) > 5000:
            text = text[:5000] + "\n...[truncated]..."

        SERVER_STATS["requests_success"] += 1
        return {
            "status_code": resp.status_code,
            "headers": dict(resp.headers),
            "body": text,
        }
    except Exception as e:
        SERVER_STATS["requests_failed"] += 1
        return {"error": str(e), "url": url}

# 
# Tools - File Operations (Sandboxed)
# 

def _safe_path(relative_path: str) -> Path:
    """Ensure path is within BASE_DIR"""
    normalized = Path(relative_path).as_posix().lstrip('/')
    full = (BASE_DIR / normalized).resolve()

    if not str(full).startswith(str(BASE_DIR.resolve())):
        raise ValueError(f"Invalid path (outside base dir): {relative_path}")

    return full

@mcp.tool()
def list_files(subdir: Optional[str] = "") -> list:
    """List files in directory (sandboxed to mcp_data/)"""
    SERVER_STATS["requests_total"] += 1
    try:
        path = _safe_path(subdir or "")
        if not path.exists():
            return []

        items = []
        for item in path.iterdir():
            items.append({
                "name": item.name,
                "is_dir": item.is_dir(),
                "size": item.stat().st_size if item.is_file() else None,
            })

        SERVER_STATS["requests_success"] += 1
        return items
    except Exception as e:
        SERVER_STATS["requests_failed"] += 1
        logger.error(f"list_files error: {e}")
        return [{"error": str(e)}]

@mcp.tool()
def read_file(path: str, max_bytes: int = 5000) -> Dict[str, Any]:
    """Read file content (sandboxed to mcp_data/)"""
    SERVER_STATS["requests_total"] += 1
    try:
        full = _safe_path(path)
        if not full.exists():
            return {"error": "file_not_found"}

        with open(full, "r", encoding="utf-8", errors="replace") as f:
            data = f.read(max_bytes + 1)

        SERVER_STATS["requests_success"] += 1
        return {
            "path": path,
            "content": data[:max_bytes],
            "truncated": len(data) > max_bytes
        }
    except Exception as e:
        SERVER_STATS["requests_failed"] += 1
        logger.error(f"read_file error: {e}")
        return {"error": str(e)}

@mcp.tool()
def write_file(path: str, content: str, overwrite: bool = True) -> Dict[str, Any]:
    """Write content to file (sandboxed to mcp_data/)"""
    SERVER_STATS["requests_total"] += 1
    try:
        full = _safe_path(path)

        if full.exists() and not overwrite:
            return {"error": "file_exists"}

        full.parent.mkdir(parents=True, exist_ok=True)

        with open(full, "w", encoding="utf-8") as f:
            f.write(content)

        SERVER_STATS["requests_success"] += 1
        return {"status": "ok", "path": path, "bytes_written": len(content)}
    except Exception as e:
        SERVER_STATS["requests_failed"] += 1
        logger.error(f"write_file error: {e}")
        return {"error": str(e)}

# 
# Main Entry Point
# 

def main():
    """Start MCP server"""
    mode = "stdio"

    # Check command line args
    if len(sys.argv) > 1 and sys.argv[1].lower() == "http":
        mode = "http"

    # Get port from environment or default
    port = int(os.environ.get("MCP_PORT", "8001"))
    host = os.environ.get("MCP_HOST", "0.0.0.0")

    if mode == "http":
        logger.info(f"= Starting MCP Server v2.0 (HTTP mode)")
        logger.info(f"   Host: {host}")
        logger.info(f"   Port: {port}")
        logger.info(f"   Health: http://{host}:{port}/health")

        # Set env vars for FastMCP
        os.environ["HOST"] = host
        os.environ["PORT"] = str(port)

        # Run with SSE transport
        mcp.run(transport="sse")
    else:
        logger.info("= Starting MCP Server v2.0 (stdio mode)")
        mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
