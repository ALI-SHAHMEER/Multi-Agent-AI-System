"""
════════════════════════════════════════════════════════════════════════
PART G  –  MCP Server Integration
Two MCP servers:
  1. Filesystem MCP – read/write local research files
  2. Google Drive MCP – access cloud-stored papers and notes

Each class follows the MCP tool-calling contract so they can be
swapped for real MCP SDK clients when running with an MCP runtime.
════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional
from config import FILESYSTEM_BASE_PATH, GOOGLE_DRIVE_FOLDER_ID


# ─── Base MCP tool interface  ─────────────────────────────────────────────────
class MCPTool:
    """Minimal contract every MCP tool must satisfy."""

    name: str = "base_tool"
    description: str = ""

    def call(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════════════════
# MCP SERVER 1 – Filesystem
# Tools: list_files, read_file, write_file, search_files
# ═══════════════════════════════════════════════════════════════════════════════
class FilesystemListTool(MCPTool):
    name = "filesystem_list"
    description = "List all research files in the local data directory."

    def call(self, directory: str = FILESYSTEM_BASE_PATH, extension: str = "") -> Dict:
        base = Path(directory)
        base.mkdir(parents=True, exist_ok=True)
        files = []
        for f in base.rglob("*"):
            if f.is_file():
                if not extension or f.suffix.lower() == extension.lower():
                    files.append({
                        "name": f.name,
                        "path": str(f),
                        "size_kb": round(f.stat().st_size / 1024, 2),
                        "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                    })
        return {"files": files, "count": len(files), "directory": str(base)}


class FilesystemReadTool(MCPTool):
    name = "filesystem_read"
    description = "Read the contents of a local research file."

    def call(self, file_path: str) -> Dict:
        path = Path(file_path)
        if not path.exists():
            return {"error": f"File not found: {file_path}"}
        try:
            content = path.read_text(encoding="utf-8")
            return {
                "file": str(path),
                "content": content,
                "size": len(content),
            }
        except Exception as e:
            return {"error": str(e)}


class FilesystemWriteTool(MCPTool):
    name = "filesystem_write"
    description = "Write or append research notes to a local file."

    def call(self, file_path: str, content: str, mode: str = "write") -> Dict:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_mode = "a" if mode == "append" else "w"
        try:
            with open(path, write_mode, encoding="utf-8") as f:
                if mode == "append":
                    f.write(f"\n\n--- {datetime.now().isoformat()} ---\n{content}")
                else:
                    f.write(content)
            return {"success": True, "file": str(path), "mode": mode}
        except Exception as e:
            return {"error": str(e)}


class FilesystemSearchTool(MCPTool):
    name = "filesystem_search"
    description = "Search for a keyword across all local research files."

    def call(self, keyword: str, directory: str = FILESYSTEM_BASE_PATH) -> Dict:
        matches = []
        base = Path(directory)
        for f in base.rglob("*.txt"):
            try:
                text = f.read_text(encoding="utf-8", errors="ignore")
                lines = [ln for ln in text.splitlines() if keyword.lower() in ln.lower()]
                if lines:
                    matches.append({"file": str(f), "matching_lines": lines[:5]})
            except Exception:
                pass
        return {"keyword": keyword, "matches": matches, "total_files_matched": len(matches)}


class FilesystemMCPServer:
    """
    Wraps all filesystem tools into a single server object.
    In a real MCP deployment this class would be the server entry point.
    """
    server_name = "filesystem-mcp"
    description = "Read, write and search local research files"

    def __init__(self):
        self._tools = {
            "list":   FilesystemListTool(),
            "read":   FilesystemReadTool(),
            "write":  FilesystemWriteTool(),
            "search": FilesystemSearchTool(),
        }

    def list_tools(self) -> List[Dict]:
        return [{"name": t.name, "description": t.description}
                for t in self._tools.values()]

    def call_tool(self, tool_name: str, **kwargs) -> Dict:
        tool = self._tools.get(tool_name)
        if not tool:
            return {"error": f"Unknown tool: {tool_name}"}
        print(f"  [Filesystem MCP] Calling {tool_name} with {kwargs}")
        return tool.call(**kwargs)

    # Convenience methods used by the agents
    def list_files(self, extension: str = "") -> Dict:
        return self.call_tool("list", extension=extension)

    def read_file(self, path: str) -> Dict:
        return self.call_tool("read", file_path=path)

    def write_file(self, path: str, content: str, mode: str = "write") -> Dict:
        return self.call_tool("write", file_path=path, content=content, mode=mode)

    def search_files(self, keyword: str) -> Dict:
        return self.call_tool("search", keyword=keyword)


# ═══════════════════════════════════════════════════════════════════════════════
# MCP SERVER 2 – Google Drive
# Tools: list_drive_files, read_drive_file, create_drive_doc
#
# NOTE: In production this uses the real MCP Google Drive server.
# Here we simulate it so the project runs without OAuth credentials,
# and provide the real connection code commented below.
# ═══════════════════════════════════════════════════════════════════════════════
SIMULATED_DRIVE_FILES = [
    {
        "id": "drive_doc_001",
        "name": "Literature Review – Transformers.gdoc",
        "type": "Google Doc",
        "modified": "2024-11-01T10:00:00Z",
        "snippet": "A comprehensive review of transformer architectures from 2017-2024, "
                   "covering BERT, GPT series, T5, and PaLM…",
    },
    {
        "id": "drive_doc_002",
        "name": "RAG Survey 2024.pdf",
        "type": "PDF",
        "modified": "2024-10-15T08:30:00Z",
        "snippet": "Survey of retrieval-augmented generation methods, covering dense retrieval, "
                   "hybrid search, and re-ranking strategies…",
    },
    {
        "id": "drive_doc_003",
        "name": "Research Notes – Multi-Agent Systems.gdoc",
        "type": "Google Doc",
        "modified": "2024-11-10T14:20:00Z",
        "snippet": "Notes on LangGraph, CrewAI, and AutoGen multi-agent frameworks…",
    },
]


class GoogleDriveMCPServer:
    """
    Simulated Google Drive MCP server.

    Real implementation uses:
        from mcp import ClientSession
        from mcp.client.sse import sse_client
        server_url = "https://mcp.googleapis.com/drive"
        # authenticate with OAuth2, then call list_resources / read_resource
    """
    server_name = "google-drive-mcp"
    description = "Access research papers and notes stored in Google Drive"

    def list_tools(self) -> List[Dict]:
        return [
            {"name": "gdrive_list",   "description": "List files in the research Drive folder"},
            {"name": "gdrive_read",   "description": "Read the content of a Drive file"},
            {"name": "gdrive_create", "description": "Create a new Google Doc for research notes"},
        ]

    def call_tool(self, tool_name: str, **kwargs) -> Dict:
        print(f"  [Google Drive MCP] Calling {tool_name} with {kwargs}")
        if tool_name == "gdrive_list":
            return self.list_files(**kwargs)
        if tool_name == "gdrive_read":
            return self.read_file(**kwargs)
        if tool_name == "gdrive_create":
            return self.create_doc(**kwargs)
        return {"error": f"Unknown tool: {tool_name}"}

    def list_files(self, query: str = "", max_results: int = 10) -> Dict:
        files = SIMULATED_DRIVE_FILES
        if query:
            files = [f for f in files if query.lower() in f["name"].lower()
                     or query.lower() in f.get("snippet", "").lower()]
        return {"files": files[:max_results], "count": len(files)}

    def read_file(self, file_id: str) -> Dict:
        for f in SIMULATED_DRIVE_FILES:
            if f["id"] == file_id:
                return {
                    "id": file_id,
                    "name": f["name"],
                    "content": f["snippet"] + "\n\n[Full content would stream here in production]",
                }
        return {"error": f"File not found: {file_id}"}

    def create_doc(self, title: str, content: str) -> Dict:
        doc_id = f"drive_new_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        return {
            "success": True,
            "doc_id": doc_id,
            "title": title,
            "url": f"https://docs.google.com/document/d/{doc_id}",
            "note": "Simulated – in production this creates a real Google Doc",
        }


# ─── MCP Layer: unified interface for agents  ────────────────────────────────
class MCPLayer:
    """
    Single entry point for all MCP servers.
    Agents call mcp.tool("server_name", "tool_name", **kwargs).
    """

    def __init__(self):
        self.filesystem = FilesystemMCPServer()
        self.google_drive = GoogleDriveMCPServer()
        self._servers = {
            "filesystem":   self.filesystem,
            "google_drive": self.google_drive,
        }

    def list_all_tools(self) -> Dict[str, List]:
        return {name: srv.list_tools() for name, srv in self._servers.items()}

    def tool(self, server: str, tool_name: str, **kwargs) -> Dict:
        srv = self._servers.get(server)
        if not srv:
            return {"error": f"Unknown server: {server}"}
        return srv.call_tool(tool_name, **kwargs)


# ─── Demo  ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mcp = MCPLayer()

    print("=== Available MCP Tools ===")
    for server, tools in mcp.list_all_tools().items():
        print(f"\n{server}:")
        for t in tools:
            print(f"  • {t['name']}: {t['description']}")

    print("\n=== Filesystem MCP Demo ===")
    # Write a sample research note
    result = mcp.tool(
        "filesystem", "write",
        file_path="./data/research_notes.txt",
        content="Transformer architecture research notes\nDate: 2024-11-15\n\nKey finding: Attention is O(n²).",
    )
    print(f"Write result: {result}")

    # List files
    result = mcp.tool("filesystem", "list", extension=".txt")
    print(f"Files: {result}")

    print("\n=== Google Drive MCP Demo ===")
    result = mcp.tool("google_drive", "gdrive_list", query="transformer")
    print(f"Drive files matching 'transformer': {json.dumps(result, indent=2)}")
