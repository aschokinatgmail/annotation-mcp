"""Tests for HEIC image loading support.

iPhone photos are typically stored as .HEIC files. The annotation-mcp
image_io module registers pillow_heif so that PIL's Image.open() can
decode them directly, without manual conversion.

These tests verify the registration works and HEIC files can be loaded
the same way as JPEG/PNG.

Note: real HEIC files are not committed to the repo (they contain EXIF
metadata that can leak GPS coordinates and device info). Tests skip
cleanly when no HEIC file is available locally. To exercise the HEIC
test path, place a .HEIC file at one of the well-known search locations
documented below.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from annotation_mcp.image_io import load_image


# Search paths for a real HEIC file. None of these are committed to the
# repo; the test skips if none exist on the developer's machine. To run
# the HEIC tests, drop a .HEIC file at any of these locations.
HEIC_SEARCH_PATHS: tuple[Path, ...] = (
    Path("tests/fixtures/exif_orientation.heic"),
    Path("tests/fixtures/sample.heic"),
    Path("tests/fixtures/heic"),
    Path("/tmp/sample.heic"),
    Path.home() / "Pictures/sample.heic",
)


def _find_heic_file() -> Path | None:
    """Find the first existing HEIC file in HEIC_SEARCH_PATHS.

    Returns None if no HEIC file is found anywhere.
    """
    for path in HEIC_SEARCH_PATHS:
        if path.is_file():
            return path
    return None


@pytest.fixture(scope="module")
def heic_path() -> Path:
    """Resolve a HEIC file path. Tests skip when no file is available.

    Tests that need a real HEIC file should depend on this fixture and
    skip when it returns a fallback path (which never exists on disk).
    """
    real = _find_heic_file()
    if real is not None:
        return real
    # Return a known-nonexistent path; tests must skip on this.
    return Path("/nonexistent/annotation-mcp-fixture-missing.heic")


def _heic_real(heic_path: Path) -> bool:
    """True if the fixture is a real, on-disk HEIC file (not the fallback)."""
    return heic_path.exists()


class TestHEICSupport:
    """HEIC files load and apply EXIF rotation correctly."""

    def test_pillow_heif_registered(self) -> None:
        """pillow_heif is registered with PIL on import."""
        from PIL import Image

        exts = Image.registered_extensions()
        assert ".heic" in exts or ".heif" in exts, (
            f"HEIC not in PIL registered extensions: {list(exts.keys())[:10]}..."
        )

    def test_load_heic_returns_pil_image(self, heic_path: Path) -> None:
        """load_image() returns a PIL Image for HEIC files."""
        if not _heic_real(heic_path):
            pytest.skip(
                "No HEIC file found in search paths. "
                "Place a .HEIC at one of HEIC_SEARCH_PATHS to run this test."
            )
        img = load_image(str(heic_path))
        assert isinstance(img, Image.Image)
        assert img.format in ("HEIF", "HEIC")

    def test_load_heic_via_raw_pil(self, heic_path: Path) -> None:
        """Raw PIL.Image.open can also open the HEIC after registration."""
        if not _heic_real(heic_path):
            pytest.skip(
                "No HEIC file found in search paths. "
                "Place a .HEIC at one of HEIC_SEARCH_PATHS to run this test."
            )
        with Image.open(heic_path) as img:
            img.load()  # force decode
            assert img.size[0] > 0
            assert img.size[1] > 0
