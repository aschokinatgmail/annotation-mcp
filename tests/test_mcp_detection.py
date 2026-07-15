"""Integration tests for the new detection tools via the MCP stdio interface.

Verifies that the tools are properly registered, the JSON-RPC request/response
format works, and the structured output matches the schema.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Generator, cast

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
SYNTHETIC_TEXT = str(FIXTURES / "synthetic_text.png")
SYNTHETIC_BARCODE = str(FIXTURES / "synthetic_ean13.png")
GARDEN_PHOTO = str(FIXTURES / "garden_photo.jpg")


class _MCPClient:
    """Minimal stdio JSON-RPC client for testing the annotation-mcp server."""

    def __init__(self) -> None:
        self.proc = subprocess.Popen(
            ["annotation-mcp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )

    def request(self, method: str, params: dict[str, object]) -> dict[str, object]:
        msg = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params,
        }
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(msg, separators=(",", ":")) + "\n")
        self.proc.stdin.flush()
        assert self.proc.stdout is not None
        line = self.proc.stdout.readline()
        return json.loads(line)  # type: ignore[no-any-return]

    def initialize(self) -> None:
        self.request(
            "initialize",
            {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1"},
            },
        )
        # Send initialized notification
        msg = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(msg, separators=(",", ":")) + "\n")
        self.proc.stdin.flush()

    def call_tool(self, name: str, arguments: dict[str, object]) -> dict[str, object]:
        return self.request("tools/call", {"name": name, "arguments": arguments})

    def list_tools(self) -> dict[str, object]:
        return self.request("tools/list", {})

    def close(self) -> None:
        self.proc.terminate()
        self.proc.wait(timeout=5)


@pytest.fixture
def mcp_client() -> "Generator[_MCPClient, None, None]":
    client = _MCPClient()
    client.initialize()
    yield client
    client.close()


class TestMCPDetectBarcodes:
    """detect_barcodes via the MCP JSON-RPC interface."""

    def test_tool_registered(self, mcp_client: _MCPClient) -> None:
        result = mcp_client.list_tools()
        assert "result" in result
        tools = result["result"]["tools"]  # type: ignore[index]
        names = [t["name"] for t in tools]  # type: ignore[union-attr]
        assert "detect_barcodes" in names

    def test_call_with_synthetic_barcode(self, mcp_client: _MCPClient) -> None:
        if not Path(SYNTHETIC_BARCODE).exists():
            pytest.skip("synthetic_ean13.png fixture not created")
        if not Path("/usr/local/lib/libzbar.dylib").exists() and not Path(
            "/opt/homebrew/lib/libzbar.dylib"
        ).exists():
            pytest.skip("zbar not installed")
        result = mcp_client.call_tool(
            "detect_barcodes", {"path": SYNTHETIC_BARCODE}
        )
        assert "result" in result
        r = cast("dict[str, object]", result["result"])
        assert r.get("isError") is False
        sc = cast("dict[str, object]", r["structuredContent"])
        assert sc["count"] >= 1  # type: ignore[operator]
        barcodes = cast("list[dict[str, object]]", sc["barcodes"])
        assert barcodes[0]["type"] == "EAN13"
        assert barcodes[0]["data"] == "4607026420155"

    def test_call_with_allowed_types(self, mcp_client: _MCPClient) -> None:
        if not Path(SYNTHETIC_BARCODE).exists():
            pytest.skip("synthetic_ean13.png fixture not created")
        if not Path("/usr/local/lib/libzbar.dylib").exists() and not Path(
            "/opt/homebrew/lib/libzbar.dylib"
        ).exists():
            pytest.skip("zbar not installed")
        result = mcp_client.call_tool(
            "detect_barcodes",
            {"path": SYNTHETIC_BARCODE, "allowed_types": ["QRCODE"]},
        )
        r = cast("dict[str, object]", result["result"])
        assert r.get("isError") is False
        sc = cast("dict[str, object]", r["structuredContent"])
        assert sc["count"] == 0

    def test_call_with_missing_file(self, mcp_client: _MCPClient) -> None:
        result = mcp_client.call_tool(
            "detect_barcodes", {"path": "/nonexistent/file.jpg"}
        )
        r = cast("dict[str, object]", result["result"])
        assert r.get("isError") is True


class TestMCPDetectTextRegions:
    """detect_text_regions via the MCP JSON-RPC interface."""

    def test_tool_registered(self, mcp_client: _MCPClient) -> None:
        result = mcp_client.list_tools()
        tools = result["result"]["tools"]  # type: ignore[index]
        names = [t["name"] for t in tools]  # type: ignore[union-attr]
        assert "detect_text_regions" in names

    def test_call_with_synthetic_text(self, mcp_client: _MCPClient) -> None:
        if not Path(SYNTHETIC_TEXT).exists():
            pytest.skip("synthetic_text.png fixture not created")
        if not shutil.which("tesseract"):
            pytest.skip("tesseract not installed")
        result = mcp_client.call_tool(
            "detect_text_regions",
            {"path": SYNTHETIC_TEXT, "lang": "eng", "detail": "word"},
        )
        r = cast("dict[str, object]", result["result"])
        assert r.get("isError") is False
        sc = cast("dict[str, object]", r["structuredContent"])
        assert sc["count"] > 0  # type: ignore[operator]
        assert sc["lang"] == "eng"
        assert sc["detail"] == "word"

    def test_call_with_invalid_detail(self, mcp_client: _MCPClient) -> None:
        if not shutil.which("tesseract"):
            pytest.skip("tesseract not installed")
        result = mcp_client.call_tool(
            "detect_text_regions", {"path": SYNTHETIC_TEXT, "detail": "garbage"}
        )
        r = cast("dict[str, object]", result["result"])
        assert r.get("isError") is True

    def test_call_with_cyrillic(self, mcp_client: _MCPClient) -> None:
        if not Path(SYNTHETIC_TEXT).exists():
            pytest.skip("synthetic_text.png fixture not created")
        if not shutil.which("tesseract"):
            pytest.skip("tesseract not installed")
        result = subprocess.run(
            ["tesseract", "--list-langs"],
            capture_output=True,
            text=True,
            check=False,
        )
        if "rus" not in result.stdout.split():
            pytest.skip("rus language pack not installed")
        result = mcp_client.call_tool(
            "detect_text_regions", {"path": SYNTHETIC_TEXT, "lang": "rus"}
        )
        r = cast("dict[str, object]", result["result"])
        assert r.get("isError") is False
        sc = cast("dict[str, object]", r["structuredContent"])
        # Cyrillic detection may or may not find our text; just verify structure
        assert "regions" in sc
        assert "count" in sc


class TestMCPCropForInspection:
    """crop_for_inspection via the MCP JSON-RPC interface."""

    def test_tool_registered(self, mcp_client: _MCPClient) -> None:
        result = mcp_client.list_tools()
        tools = result["result"]["tools"]  # type: ignore[index]
        names = [t["name"] for t in tools]  # type: ignore[union-attr]
        assert "crop_for_inspection" in names

    def test_call_basic_crop(self, mcp_client: _MCPClient, tmp_path: Path) -> None:
        from PIL import Image

        src = tmp_path / "src.png"
        Image.new("RGB", (400, 300), "white").save(src)
        out = tmp_path / "out.png"
        result = mcp_client.call_tool(
            "crop_for_inspection",
            {
                "path": str(src),
                "bbox": [100, 50, 300, 250],
                "output_path": str(out),
            },
        )
        r = cast("dict[str, object]", result["result"])
        assert r.get("isError") is False
        sc = cast("dict[str, object]", r["structuredContent"])
        crop = cast("dict[str, object]", sc["crop"])
        assert crop["width"] == 200
        assert crop["height"] == 200
        assert Path(cast("str", crop["output_path"])).exists()

    def test_call_normalized_crop(
        self, mcp_client: _MCPClient, tmp_path: Path
    ) -> None:
        from PIL import Image

        src = tmp_path / "src.png"
        Image.new("RGB", (400, 300), "white").save(src)
        result = mcp_client.call_tool(
            "crop_for_inspection",
            {
                "path": str(src),
                "bbox": [0.25, 0.25, 0.75, 0.75],
                "normalized": True,
            },
        )
        r = cast("dict[str, object]", result["result"])
        assert r.get("isError") is False
        sc = cast("dict[str, object]", r["structuredContent"])
        crop = cast("dict[str, object]", sc["crop"])
        # 25-75% of 400x300 = 100,75 to 300,225 -> 200x150
        assert crop["width"] == 200
        assert crop["height"] == 150

    def test_call_missing_file(self, mcp_client: _MCPClient, tmp_path: Path) -> None:
        out = tmp_path / "out.png"
        result = mcp_client.call_tool(
            "crop_for_inspection",
            {
                "path": "/nonexistent/file.png",
                "bbox": [0, 0, 100, 100],
                "output_path": str(out),
            },
        )
        r = cast("dict[str, object]", result["result"])
        assert r.get("isError") is True
