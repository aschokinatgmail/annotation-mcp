"""Functional tests for the MCP server — handshake + tool listing."""
from __future__ import annotations
import base64
import json
import os
import subprocess
import sys
import tempfile
import time
import pytest
from PIL import Image
from annotation_mcp.image_io import is_png_data

SERVER_SCRIPT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src", "annotation_mcp", "server.py")
)


@pytest.fixture
def test_jpeg() -> str:
    """Create a test JPEG image and return its path."""
    img = Image.new("RGB", (320, 240), color=(200, 200, 200))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 150, 150], fill=(100, 100, 100))
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img.save(tmp.name, format="JPEG")
    tmp.close()
    return tmp.name


class MCPServerClient:
    """Helper to interact with the MCP server over stdio."""

    def __init__(self) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, str(SERVER_SCRIPT)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(0.3)

    def send(self, request: dict[str, object]) -> dict[str, object]:
        """Send a JSON-RPC request and return the response."""
        line = json.dumps(request) + "\n"
        if self.proc.stdin is None:
            msg = "Server stdin is None"
            raise RuntimeError(msg)
        self.proc.stdin.write(line.encode("utf-8"))
        self.proc.stdin.flush()
        if self.proc.stdout is None:
            msg = "Server stdout is None"
            raise RuntimeError(msg)
        response_line = self.proc.stdout.readline()
        result: dict[str, object] = json.loads(response_line.decode("utf-8"))
        return result

    def initialize(self) -> dict[str, object]:
        """Initialize the MCP server."""
        return self.send({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05", "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0"},
            },
        })

    def send_initialized(self) -> None:
        """Send the initialized notification (no response expected)."""
        line = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n"
        if self.proc.stdin is None:
            msg = "Server stdin is None"
            raise RuntimeError(msg)
        self.proc.stdin.write(line.encode("utf-8"))
        self.proc.stdin.flush()

    def list_tools(self) -> list[dict[str, object]]:
        """List available tools."""
        response = self.send({
            "jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {},
        })
        result = response.get("result")
        assert isinstance(result, dict)
        tools = result.get("tools", [])
        assert isinstance(tools, list)
        return tools

    def call_tool(self, name: str, arguments: dict[str, object]) -> dict[str, object]:
        """Call a tool."""
        return self.send({
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        })

    def close(self) -> None:
        """Terminate the server."""
        self.proc.terminate()
        self.proc.wait(timeout=5)


class TestMCPHandshake:
    """Tests for MCP protocol handshake and tool listing."""

    def test_initialize_and_list_tools(self) -> None:
        """Given server started, when initialized and tools/list called, then returns 5 tools."""
        client = MCPServerClient()
        try:
            init_response = client.initialize()
            assert init_response.get("id") == 1
            assert "result" in init_response
            result = init_response["result"]
            assert isinstance(result, dict)
            assert "protocolVersion" in result
            assert "capabilities" in result
            assert "serverInfo" in result
            client.send_initialized()
            tools = client.list_tools()
            tool_names = [str(t.get("name", "")) for t in tools]
            assert "get_image_info" in tool_names
            assert "annotate" in tool_names
            assert "draw_bounding_box" in tool_names
            assert "highlight_region" in tool_names
            assert "draw_numbered_callouts" in tool_names
            assert "detect_barcodes" in tool_names
            assert "detect_text_regions" in tool_names
            assert "crop_for_inspection" in tool_names
            assert len(tool_names) == 8
        finally:
            client.close()
