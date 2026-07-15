"""Text region detection via Tesseract OCR.

Returns every detected text region with its bbox, text content, and
Tesseract confidence score. This is the workhorse for "find the brand
title" / "find the dosage table" workflows — the AI gets structured
text+bbox data and just picks the region whose text matches the
semantic label.

Design notes:
    - Lazy import pytesseract so the MCP server works without it.
    - Three preprocessing modes:
        * ``none``: pass the image as-is (fastest, works on clean scans)
        * ``clahe``: Contrast Limited Adaptive Histogram Equalization
          (good for photos with uneven lighting)
        * ``adaptive``: adaptive thresholding (good for high-contrast
          text on flat backgrounds, e.g. signs)
        * ``otsu``: Otsu global thresholding (fast, good for clean b&w)
    - All preprocessing requires opencv-python. If opencv is missing,
        preprocessing falls back to grayscale + PIL autocontrast.
    - Languages: pass any combination accepted by tesseract (e.g. ``eng+rus``).
        The Tesseract language packs must be installed separately
        (``brew install tesseract-lang``).
    - Two ``detail`` levels:
        * ``word``: each match is one word with its tight bbox
        * ``line``: each match is one line with a line-level bbox
        * ``block``: each match is a text block (paragraph) bbox
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from PIL import Image, ImageOps

from annotation_mcp.detection.garbage_filter import looks_like_garbage
from annotation_mcp.image_io import load_image

if TYPE_CHECKING:
    pass

DetailLevel = Literal["word", "line", "block", "paragraph"]
PreprocessMode = Literal["none", "clahe", "adaptive", "otsu"]


@dataclass(frozen=True, slots=True)
class TextMatch:
    """A single detected text region."""

    text: str
    confidence: float  # 0-100 (Tesseract's native scale)
    bbox_pixels: tuple[int, int, int, int]  # (x, y, x2, y2)
    bbox_normalized: tuple[float, float, float, float]
    block_num: int
    line_num: int
    level: str  # "word" | "line" | "block" | "paragraph"

    def to_dict(self, image_width: int, image_height: int) -> dict[str, object]:
        """Serialize to a JSON-safe dict."""
        return {
            "text": self.text,
            "confidence": self.confidence,
            "bbox_pixels": list(self.bbox_pixels),
            "bbox_normalized": list(self.bbox_normalized),
            "block": self.block_num,
            "line": self.line_num,
            "level": self.level,
            "image_width": image_width,
            "image_height": image_height,
        }


class TextDetectionError(RuntimeError):
    """Raised when text detection cannot run."""


def _preprocess_image(
    img: Image.Image,
    mode: PreprocessMode,
) -> Image.Image:
    """Apply preprocessing to a PIL image for better OCR accuracy.

    Falls back to PIL-only operations if opencv is unavailable.
    """
    if mode == "none":
        return img

    # Convert to grayscale first (most preprocessing works on single channel)
    gray = img.convert("L")

    if mode == "clahe":
        try:
            import cv2
            import numpy as np
        except ImportError:
            # Fall back to PIL autocontrast
            return ImageOps.autocontrast(gray)

        arr = np.asarray(gray)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        out = clahe.apply(arr)
        return Image.fromarray(out)

    if mode == "adaptive":
        try:
            import cv2
            import numpy as np
        except ImportError:
            return ImageOps.autocontrast(gray)

        arr = np.asarray(gray)
        # Block size must be odd and > 1; 31 is a safe default for text.
        out = cv2.adaptiveThreshold(
            arr,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            15,
        )
        return Image.fromarray(out)

    if mode == "otsu":
        try:
            import cv2
            import numpy as np
        except ImportError:
            return ImageOps.autocontrast(gray)

        arr = np.asarray(gray)
        _, out = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return Image.fromarray(out)

    # Defensive: should be unreachable because PreprocessMode is a closed
    # Literal type, but keep the raise so that any future addition to the
    # Literal (without updating this function) fails loudly rather than
    # silently returning the unprocessed image.
    msg = f"Unknown preprocess mode: {mode!r}"
    raise ValueError(msg)


def _safe_str(value: object) -> str:
    """Convert a Tesseract data dict value to string safely."""
    if value is None:
        return ""
    return str(value)


def _safe_int(value: object, default: int = 0) -> int:
    """Convert a Tesseract data dict value to int safely."""
    if value is None:
        return default
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def _safe_float(value: object, default: float = 0.0) -> float:
    """Convert a Tesseract data dict value to float safely."""
    if value is None:
        return default
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def _run_ocr(
    preprocessed: Image.Image,
    *,
    lang: str,
    psm: int | None,
    pytesseract_module: Any,
) -> dict[str, list[object]]:
    """Call pytesseract.image_to_data and return its raw dict.

    Centralised so the fallback path can call it with different PSM values.
    The pytesseract module is passed in to avoid repeated imports; the type
    hint is intentionally broad (Any) because pytesseract ships no type stubs.
    """
    config_parts: list[str] = []
    if psm is not None:
        config_parts.append(f"--psm {psm}")
    config: str = " ".join(config_parts) if config_parts else ""

    raw_data = pytesseract_module.image_to_data(
        preprocessed,
        lang=lang,
        output_type=pytesseract_module.Output.DICT,
        config=config,
    )
    return cast("dict[str, list[object]]", raw_data)


def detect_text_regions(  # noqa: C901, PLR0913
    path: str,
    *,
    lang: str = "eng",
    detail: DetailLevel = "word",
    preprocess: PreprocessMode = "clahe",
    min_confidence: float = 0.0,
    include_empty: bool = False,
    psm: int | None = None,
    filter_garbage: bool = True,
) -> list[TextMatch]:
    """Detect text regions in an image using Tesseract OCR.

    Args:
        path: Absolute path to image file.
        lang: Tesseract language code(s), e.g. ``"eng"``, ``"rus"``, ``"eng+rus"``.
        detail: Granularity of returned regions.
        preprocess: Image preprocessing mode (see module docstring).
        min_confidence: Discard regions with confidence below this (Tesseract scale 0-100).
        include_empty: If True, include regions with empty text (e.g. blocks with no
            text yet). Default False skips them.
        psm: Optional Tesseract Page Segmentation Mode (1-13). 6 = uniform block,
            11 = sparse text, 12 = sparse with OSD. None = tesseract default.
        filter_garbage: If True (default), apply a heuristic filter that drops
            OCR-noise regions (high-confidence nonsense like 'VLOAOL', 'yOLO',
            'aS', '||||'). See :mod:`annotation_mcp.detection.garbage_filter`.

    Returns:
        List of :class:`TextMatch` with pixel and normalized bounding boxes,
        sorted top-to-bottom, left-to-right.

    Raises:
        TextDetectionError: If pytesseract/tesseract is not installed,
            the image cannot be loaded, or Tesseract itself fails.
    """
    try:
        import pytesseract
    except ImportError as e:
        raise TextDetectionError(
            "pytesseract not installed."
            + " Install with: pip install pytesseract;"
            + " and the tesseract binary: brew install tesseract tesseract-lang"
        ) from e

    try:
        img = load_image(path)
    except (FileNotFoundError, OSError) as e:
        raise TextDetectionError(f"Cannot load image: {e}") from e

    width, height = img.size
    preprocessed = _preprocess_image(img, preprocess)

    try:
        data = _run_ocr(preprocessed, lang=lang, psm=psm, pytesseract_module=pytesseract)
    except pytesseract.TesseractError as e:
        raise TextDetectionError(f"Tesseract OCR failed: {e}") from e
    except RuntimeError as e:
        # pytesseract raises RuntimeError if tesseract binary is missing
        raise TextDetectionError(
            "tesseract binary not found."
            + " Install with: brew install tesseract tesseract-lang"
        ) from e

    # Fallback: if user didn't specify PSM and we got 0 regions, retry with
    # psm=6 (uniform block of text) which works for the majority of images
    # where the default page-segmentation mode silently returns nothing.
    if psm is None:
        if not _data_has_word_regions(data):
            try:
                data = _run_ocr(
                    preprocessed, lang=lang, psm=6, pytesseract_module=pytesseract
                )
            except (pytesseract.TesseractError, RuntimeError):
                pass  # Fallback is best-effort; keep the empty original

    # For line/block/paragraph detail, we collapse word-level data into larger regions.
    if detail in ("line", "block", "paragraph"):
        return _collapse_to_level(
            data, detail, width, height, min_confidence, include_empty
        )

    # Word-level: filter and convert.
    matches = _build_word_matches(
        data, width, height, min_confidence, include_empty, filter_garbage
    )
    matches.sort(key=lambda m: (m.bbox_pixels[1], m.bbox_pixels[0]))
    return matches


def _data_has_word_regions(data: dict[str, list[object]]) -> bool:
    """Return True if the Tesseract output has any non-empty word-level rows."""
    text_list = data.get("text", [])
    for raw in text_list:
        if str(raw).strip():
            return True
    return False


def _build_word_matches(
    data: dict[str, list[object]],
    width: int,
    height: int,
    min_confidence: float,
    include_empty: bool,
    filter_garbage: bool,
) -> list[TextMatch]:
    """Build word-level TextMatch objects from raw Tesseract data."""
    text_list = data.get("text", [])
    n = len(text_list)
    if n == 0:
        return []

    conf_list = data.get("conf", [0] * n)
    left_list = data.get("left", [0] * n)
    top_list = data.get("top", [0] * n)
    width_list = data.get("width", [0] * n)
    height_list = data.get("height", [0] * n)
    level_list = data.get("level", [5] * n)
    block_list = data.get("block_num", [0] * n)
    line_list = data.get("line_num", [0] * n)

    matches: list[TextMatch] = []
    for i in range(n):
        text = _safe_str(text_list[i]).strip()
        conf_value = _safe_float(conf_list[i] if i < len(conf_list) else 0)

        if conf_value < min_confidence:
            continue
        if not include_empty and not text:
            continue

        level_num = _safe_int(level_list[i] if i < len(level_list) else 5, default=5)
        # Tesseract levels: 1=page, 2=block, 3=paragraph, 4=line, 5=word
        level_name = {
            1: "page",
            2: "block",
            3: "paragraph",
            4: "line",
            5: "word",
        }.get(level_num, "word")

        if level_name != "word":
            continue

        if filter_garbage and looks_like_garbage(text):
            continue

        left = _safe_int(left_list[i] if i < len(left_list) else 0)
        top = _safe_int(top_list[i] if i < len(top_list) else 0)
        w = _safe_int(width_list[i] if i < len(width_list) else 0)
        h = _safe_int(height_list[i] if i < len(height_list) else 0)
        bbox_pixels = (left, top, left + w, top + h)
        bbox_normalized = (
            left / width,
            top / height,
            (left + w) / width,
            (top + h) / height,
        )

        block_num = _safe_int(block_list[i] if i < len(block_list) else 0)
        line_num = _safe_int(line_list[i] if i < len(line_list) else 0)

        matches.append(
            TextMatch(
                text=text,
                confidence=conf_value,
                bbox_pixels=bbox_pixels,
                bbox_normalized=bbox_normalized,
                block_num=block_num,
                line_num=line_num,
                level=level_name,
            )
        )

    return matches


def _collapse_to_level(
    data: dict[str, list[object]],
    target: DetailLevel,
    width: int,
    height: int,
    min_confidence: float,
    include_empty: bool,
) -> list[TextMatch]:
    """Collapse Tesseract word-level data into line/block/paragraph level matches."""
    text_list = data.get("text", [])
    n = len(text_list)
    if n == 0:
        return []

    conf_list = data.get("conf", [0] * n)
    left_list = data.get("left", [0] * n)
    top_list = data.get("top", [0] * n)
    width_list = data.get("width", [0] * n)
    height_list = data.get("height", [0] * n)
    level_list = data.get("level", [5] * n)
    page_list = data.get("page_num", [1] * n)
    block_list = data.get("block_num", [0] * n)
    par_list = data.get("par_num", [0] * n)
    line_list = data.get("line_num", [0] * n)

    # Internal accumulator type
    @dataclass
    class _Group:
        words: list[str]
        confs: list[float]
        left: int
        top: int
        right: int
        bottom: int
        block_num: int
        line_num: int

    groups: dict[tuple[int, int, int, int], _Group] = {}

    for i in range(n):
        level_num = _safe_int(level_list[i] if i < len(level_list) else 5, default=5)
        if level_num != 5:  # only process words
            continue

        text = _safe_str(text_list[i]).strip()
        conf_value = _safe_float(conf_list[i] if i < len(conf_list) else 0)

        if conf_value < min_confidence:
            continue
        if not include_empty and not text:
            continue

        page_num = _safe_int(page_list[i] if i < len(page_list) else 1, default=1)
        block_num = _safe_int(block_list[i] if i < len(block_list) else 0)
        par_num = _safe_int(par_list[i] if i < len(par_list) else 0)
        line_num = _safe_int(line_list[i] if i < len(line_list) else 0)

        match target:
            case "line":
                key = (page_num, block_num, par_num, line_num)
            case "block":
                key = (page_num, block_num, -1, -1)
            case "paragraph":
                key = (page_num, block_num, par_num, -1)
            case _:
                key = (page_num, block_num, par_num, -1)

        left = _safe_int(left_list[i] if i < len(left_list) else 0)
        top = _safe_int(top_list[i] if i < len(top_list) else 0)
        w = _safe_int(width_list[i] if i < len(width_list) else 0)
        h = _safe_int(height_list[i] if i < len(height_list) else 0)

        existing = groups.get(key)
        if existing is None:
            groups[key] = _Group(
                words=[text],
                confs=[conf_value],
                left=left,
                top=top,
                right=left + w,
                bottom=top + h,
                block_num=block_num,
                line_num=line_num,
            )
        else:
            existing.words.append(text)
            existing.confs.append(conf_value)
            existing.left = min(existing.left, left)
            existing.top = min(existing.top, top)
            existing.right = max(existing.right, left + w)
            existing.bottom = max(existing.bottom, top + h)

    collapsed: list[TextMatch] = []
    for g in groups.values():
        bbox_pixels = (g.left, g.top, g.right, g.bottom)
        bbox_normalized = (
            g.left / width,
            g.top / height,
            g.right / width,
            g.bottom / height,
        )
        avg_conf = sum(g.confs) / len(g.confs) if g.confs else 0.0
        collapsed.append(
            TextMatch(
                text=" ".join(g.words),
                confidence=avg_conf,
                bbox_pixels=bbox_pixels,
                bbox_normalized=bbox_normalized,
                block_num=g.block_num,
                line_num=g.line_num,
                level=target,
            )
        )

    collapsed.sort(key=lambda m: (m.bbox_pixels[1], m.bbox_pixels[0]))
    return collapsed
