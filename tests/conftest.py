"""Pytest configuration and synthetic fixture generation.

Generates all synthetic test fixtures at session start so the repo
doesn't commit any image files. Real-world fixtures (garden_photo,
HEIC files) are optional — tests skip cleanly when absent.
"""
from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FIXTURES = Path(__file__).parent / "fixtures"


def _get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try to load a TrueType font; fall back to PIL default."""
    for fp in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\segoeui.ttf",
    ):
        if Path(fp).exists():
            try:
                return ImageFont.truetype(fp, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _make_synthetic_text(path: Path) -> None:
    img = Image.new("RGB", (1200, 600), "white")
    draw = ImageDraw.Draw(img)
    draw.text((450, 50), "BRAND TITLE", fill="black", font=_get_font(48))
    draw.text((50, 200), "This is the first line of body text.", fill="black", font=_get_font(24))
    draw.text((50, 250), "This is the second line of body text.", fill="black", font=_get_font(24))
    draw.text((50, 300), "This is the third line of body text.", fill="black", font=_get_font(24))
    draw.text((50, 400), "ФИТОСПОРИН", fill="black", font=_get_font(28))
    img.save(path)


def _make_synthetic_noisy_text(path: Path) -> None:
    random.seed(42)
    img = Image.new("RGB", (1200, 600), "white")
    draw = ImageDraw.Draw(img)
    draw.text((100, 100), "PRODUCT NAME", fill="black", font=_get_font(48))
    draw.text((100, 250), "Ingredients: Water, Salt", fill="black", font=_get_font(24))
    draw.text((100, 300), "Net weight: 500g", fill="black", font=_get_font(24))
    pixels = img.load()
    if pixels is not None:
        for _ in range(2000):
            x = random.randint(0, 1199)
            y = random.randint(0, 599)
            pixels[x, y] = (100, 100, 100)
    img.save(path)


def _make_synthetic_ean13(path: Path) -> None:
    L_CODE = {
        "0": "0001101", "1": "0011001", "2": "0010011", "3": "0111101",
        "4": "0100011", "5": "0110001", "6": "0101111", "7": "0111011",
        "8": "0110111", "9": "0001011",
    }
    G_CODE = {
        "0": "0100111", "1": "0110011", "2": "0011011", "3": "0100001",
        "4": "0011101", "5": "0111001", "6": "0000101", "7": "0010001",
        "8": "0001001", "9": "0010111",
    }
    R_CODE = {
        "0": "1110010", "1": "1100110", "2": "1101100", "3": "1000010",
        "4": "1011100", "5": "1001110", "6": "1010000", "7": "1000100",
        "8": "1001000", "9": "1110100",
    }
    PARITY = {
        "0": "LLLLLL", "1": "LLGLGG", "2": "LLGGLG", "3": "LLGGGL",
        "4": "LGLLGG", "5": "LGGLLG", "6": "LGGGLL", "7": "LGLGLG",
        "8": "LGLGGL", "9": "LGGLGL",
    }
    digits = "4607026420155"
    parity = PARITY[digits[0]]
    M = 3
    quiet = 9
    total = 3 + 42 + 5 + 42 + 3 + 2 * quiet
    img = Image.new("RGB", (total * M, 60 * M), "white")
    draw = ImageDraw.Draw(img)
    pos = quiet

    def dm(pat: str) -> None:
        nonlocal pos
        for ch in pat:
            if ch == "1":
                draw.rectangle(
                    [pos * M, 0, (pos + 1) * M - 1, img.height - 1],
                    fill="black",
                )
            pos += 1

    dm("101")
    for i, d in enumerate(digits[1:7]):
        dm(L_CODE[d] if parity[i] == "L" else G_CODE[d])
    dm("01010")
    for d in digits[7:]:
        dm(R_CODE[d])
    dm("101")
    img.save(path)


def _make_exif_orientation6(path: Path) -> None:
    img = Image.new("RGB", (300, 200), (100, 150, 255))
    draw = ImageDraw.Draw(img)
    draw.text((60, 80), "TOP", fill="red", font=_get_font(36))
    draw.rectangle([0, 0, 40, 40], fill="red")
    exif = img.getexif()
    exif[0x0112] = 6
    img.save(path, exif=exif.tobytes())


_GENERATORS = {
    "synthetic_text.png": _make_synthetic_text,
    "synthetic_noisy_text.png": _make_synthetic_noisy_text,
    "synthetic_ean13.png": _make_synthetic_ean13,
    "exif_orientation6.jpg": _make_exif_orientation6,
}


def pytest_configure() -> None:
    """Generate all synthetic fixtures before tests collect."""
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for name, gen in _GENERATORS.items():
        p = FIXTURES / name
        if not p.exists():
            gen(p)
