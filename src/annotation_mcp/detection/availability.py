"""Runtime detection of optional dependencies (tesseract, zbar, opencv).

These checks are cheap (single subprocess call or module import) and let
the MCP tools return actionable error messages when system libraries are
missing, rather than crashing with a confusing stack trace.

We do NOT cache results because the install state can change during a
session, and the cost is negligible.
"""
from __future__ import annotations

import ctypes.util
import importlib.util
import shutil
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DependencyStatus:
    """Status of one optional dependency."""

    name: str
    available: bool
    detail: str
    install_hint: str | None = None


def _tesseract_install_hint(action: str = "Install") -> str:
    if sys.platform == "darwin":
        return f"{action} with: brew install tesseract tesseract-lang"
    if sys.platform == "linux":
        return f"{action} with: apt install tesseract-ocr tesseract-ocr-{{lang}}"
    if sys.platform == "win32":
        return f"{action} with: choco install tesseract"
    return f"{action} tesseract for your platform"


def check_tesseract() -> DependencyStatus:
    """Check if tesseract binary is installed and get version + languages."""
    tesseract_path = shutil.which("tesseract")
    if not tesseract_path:
        return DependencyStatus(
            name="tesseract",
            available=False,
            detail="tesseract binary not found on PATH",
            install_hint=_tesseract_install_hint(),
        )

    try:
        result = subprocess.run(
            ["tesseract", "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        return DependencyStatus(
            name="tesseract",
            available=False,
            detail=f"tesseract failed to execute: {e}",
            install_hint=_tesseract_install_hint("Reinstall"),
        )

    if result.returncode != 0:
        return DependencyStatus(
            name="tesseract",
            available=False,
            detail=f"tesseract --version returned {result.returncode}",
            install_hint=_tesseract_install_hint("Reinstall"),
        )

    # Parse first line of stdout (e.g. "tesseract 5.5.2")
    first_line = (result.stdout or "").splitlines()[0] if result.stdout else ""
    return DependencyStatus(name="tesseract", available=True, detail=first_line)


def check_tesseract_languages(*required: str) -> tuple[bool, list[str]]:
    """Check if all required Tesseract language packs are installed.

    Args:
        *required: Language codes (e.g. "eng", "rus", "deu").

    Returns:
        (all_present, list_of_missing_languages)
    """
    tesseract_path = shutil.which("tesseract")
    if not tesseract_path:
        return False, list(required)

    try:
        result = subprocess.run(
            ["tesseract", "--list-langs"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False, list(required)

    if result.returncode != 0:
        return False, list(required)

    available = set((result.stdout or "").split())
    missing = [lang for lang in required if lang not in available]
    return not missing, missing


def check_zbar_library() -> DependencyStatus:
    """Check if zbar shared library is present AND loadable by pyzbar.

    On some platforms (especially macOS CI), the dylib file exists but
    the dynamic linker can't find it at runtime. We verify both the
    file existence AND that pyzbar can actually import and load it.
    """
    if importlib.util.find_spec("pyzbar") is None:
        return DependencyStatus(
            name="zbar",
            available=False,
            detail="pyzbar Python module not installed",
            install_hint="Install with: pip install pyzbar",
        )

    found_path = ctypes.util.find_library("zbar")
    if not found_path:
        for candidate in (
            "/opt/homebrew/lib/libzbar.dylib",
            "/opt/homebrew/opt/zbar/lib/libzbar.dylib",
            "/usr/local/lib/libzbar.dylib",
            "/usr/local/opt/zbar/lib/libzbar.dylib",
            "/usr/lib/libzbar.so",
            "/usr/lib/x86_64-linux-gnu/libzbar.so",
            "/usr/lib/x86_64-linux-gnu/libzbar.so.0",
            "/usr/lib/aarch64-linux-gnu/libzbar.so",
            "/usr/lib/aarch64-linux-gnu/libzbar.so.0",
        ):
            if _path_exists(candidate):
                found_path = candidate
                break

    try:
        from pyzbar.pyzbar import decode, ZBarSymbol  # noqa: F401
    except Exception as e:
        hint = "brew install zbar (macOS)"
        if found_path:
            hint += f" — library at {found_path} but import failed: {e}"
        else:
            hint += " or apt install libzbar0 (Linux)"
        return DependencyStatus(
            name="zbar",
            available=False,
            detail=f"pyzbar import failed: {e}",
            install_hint=hint,
        )

    if found_path:
        return DependencyStatus(name="zbar", available=True, detail=f"loaded from {found_path}")
    return DependencyStatus(name="zbar", available=True, detail="pyzbar importable")


def check_opencv() -> DependencyStatus:
    """Check if opencv-python is importable."""
    if importlib.util.find_spec("cv2") is None:
        return DependencyStatus(
            name="opencv-python",
            available=False,
            detail="cv2 module not installed",
            install_hint="Install with: pip install opencv-python-headless",
        )
    return DependencyStatus(name="opencv-python", available=True, detail="cv2 importable")


def _path_exists(path: str) -> bool:
    from pathlib import Path

    return Path(path).exists()
