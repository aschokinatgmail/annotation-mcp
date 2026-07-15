"""MCP tool handlers for object detection and region cropping.

These wrap the detection modules in the standardized handler/result
pattern used by the rest of the MCP server, and return a JSON-safe
structuredContent dict (not raw images — the detection tools don't
produce images).
"""
from __future__ import annotations

from pathlib import Path

from mcp.types import CallToolResult, TextContent

from annotation_mcp.detection.availability import (
    check_tesseract,
    check_tesseract_languages,
    check_zbar_library,
)
from annotation_mcp.detection.barcode import (
    BarcodeDetectionError,
    BarcodeMatch,
    detect_barcodes,
)
from annotation_mcp.detection.crop import CropError, CropResult, crop_region
from annotation_mcp.detection.ocr import (
    DetailLevel,
    PreprocessMode,
    TextDetectionError,
    TextMatch,
    detect_text_regions,
)
from annotation_mcp.image_io import get_image_info


def _error_result(message: str) -> CallToolResult:
    """Build a CallToolResult for a non-fatal error (e.g. missing deps)."""
    return CallToolResult(
        content=[TextContent(type="text", text=message)],
        isError=True,
    )


def handle_detect_barcodes(
    path: str,
    *,
    min_confidence: float = 0.0,
    allowed_types: list[str] | None = None,
) -> CallToolResult:
    """Handle detect_barcodes tool call.

    Args:
        path: Absolute path to image file.
        min_confidence: Discard matches with confidence below this (0-1).
        allowed_types: Restrict to specific barcode type names (e.g. ["EAN13", "QRCODE"]).

    Returns:
        CallToolResult with structuredContent containing detected barcodes.
    """
    zbar_status = check_zbar_library()
    if not zbar_status.available:
        return _error_result(
            f"zbar/pyzbar not available: {zbar_status.detail}. "
            + f"Install: {zbar_status.install_hint}"
        )

    if not Path(path).is_file():
        return _error_result(f"File not found: {path}")

    try:
        info = get_image_info(path)
    except (FileNotFoundError, OSError) as e:
        return _error_result(f"Cannot read image: {e}")

    try:
        matches: list[BarcodeMatch] = detect_barcodes(
            path,
            min_confidence=min_confidence,
            allowed_types=allowed_types,
        )
    except BarcodeDetectionError as e:
        return _error_result(str(e))

    structured: dict[str, object] = {
        "image_path": path,
        "image_width": info.width,
        "image_height": info.height,
        "count": len(matches),
        "barcodes": [
            m.to_dict(info.width, info.height)
            for m in matches
        ],
    }
    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=f"Detected {len(matches)} barcode(s) in {path}",
            )
        ],
        structuredContent=structured,
        isError=False,
    )


def handle_detect_text_regions(
    path: str,
    *,
    lang: str = "eng",
    detail: str = "word",
    preprocess: str = "clahe",
    min_confidence: float = 0.0,
    psm: int | None = None,
    filter_garbage: bool = True,
    crop_regions: bool = False,
    crop_padding: float = 0.1,
    crop_output_dir: str | None = None,
) -> CallToolResult:
    """Handle detect_text_regions tool call.

    Args:
        path: Absolute path to image file.
        lang: Tesseract language code(s), e.g. "eng", "rus", "eng+rus".
        detail: Granularity: "word" | "line" | "block" | "paragraph".
        preprocess: Preprocessing mode: "none" | "clahe" | "adaptive" | "otsu".
        min_confidence: Discard regions with confidence below this (0-100).
        psm: Optional Tesseract Page Segmentation Mode (1-13).
        filter_garbage: If True (default), drop OCR-noise regions
            (high-confidence nonsense like 'VLOAOL', 'yOLO'). See
            ``annotation_mcp.detection.garbage_filter``.
        crop_regions: If True, also save cropped images of each detected
            text region to disk and include the paths in the result.
            Useful for re-recognising the text with a vision model when
            tesseract produces low-quality output on hard images.
        crop_padding: Fractional padding to add around each crop (0.1 = 10%).
        crop_output_dir: Directory to save crops. If None and crop_regions
            is True, a temp directory is used.

    Returns:
        CallToolResult with structuredContent containing detected text regions.
        If crop_regions=True, also includes ``region_crops``: a list of
        {region_index, path, bbox_pixels} entries pointing to the saved
        crops so the caller can re-recognise them with a vision model.
    """
    tess_status = check_tesseract()
    if not tess_status.available:
        return _error_result(
            f"tesseract not available: {tess_status.detail}. "
            + f"Install: {tess_status.install_hint}"
        )

    # Verify each requested language is installed
    requested_langs = [code.strip() for code in lang.split("+") if code.strip()]
    all_present, missing = check_tesseract_languages(*requested_langs)
    if not all_present:
        return _error_result(
            f"Tesseract language pack(s) not installed: {missing}. "
            + "Install with: brew install tesseract-lang"
        )

    if not Path(path).is_file():
        return _error_result(f"File not found: {path}")

    try:
        info = get_image_info(path)
    except (FileNotFoundError, OSError) as e:
        return _error_result(f"Cannot read image: {e}")

    # Validate enum-like parameters
    valid_details: tuple[DetailLevel, ...] = ("word", "line", "block", "paragraph")
    if detail not in valid_details:
        return _error_result(
            f"Invalid detail '{detail}'. Must be one of: {', '.join(valid_details)}"
        )
    valid_preprocess: tuple[PreprocessMode, ...] = ("none", "clahe", "adaptive", "otsu")
    if preprocess not in valid_preprocess:
        return _error_result(
            f"Invalid preprocess '{preprocess}'. Must be one of: {', '.join(valid_preprocess)}"
        )

    try:
        matches: list[TextMatch] = detect_text_regions(
            path,
            lang=lang,
            detail=detail,
            preprocess=preprocess,
            min_confidence=min_confidence,
            psm=psm,
            filter_garbage=filter_garbage,
        )
    except TextDetectionError as e:
        return _error_result(str(e))

    structured: dict[str, object] = {
        "image_path": path,
        "image_width": info.width,
        "image_height": info.height,
        "lang": lang,
        "detail": detail,
        "preprocess": preprocess,
        "filter_garbage": filter_garbage,
        "count": len(matches),
        "regions": [
            m.to_dict(info.width, info.height)
            for m in matches
        ],
    }

    if crop_regions and matches:
        from annotation_mcp.detection.crop import crop_region

        if crop_output_dir is None:
            import tempfile

            crop_output_dir = tempfile.mkdtemp(prefix="annotation_mcp_crops_")
        crop_dir = Path(crop_output_dir)
        crop_dir.mkdir(parents=True, exist_ok=True)

        region_crops: list[dict[str, object]] = []
        for i, m in enumerate(matches):
            x1, y1, x2, y2 = m.bbox_pixels
            # Generate a descriptive filename
            text_slug = "".join(
                c if c.isalnum() else "_" for c in (m.text or "region")[:20]
            ).strip("_") or "region"
            crop_path = crop_dir / f"region_{i:03d}_{text_slug}.jpg"
            try:
                result = crop_region(
                    path,
                    (x1, y1, x2, y2),
                    output_path=str(crop_path),
                    normalized=False,
                    padding=crop_padding,
                )
                region_crops.append(
                    {
                        "region_index": i,
                        "text": m.text,
                        "confidence": m.confidence,
                        "bbox_pixels": list(m.bbox_pixels),
                        "crop_path": result.output_path,
                        "crop_size": [result.width, result.height],
                    }
                )
            except Exception as e:
                region_crops.append(
                    {
                        "region_index": i,
                        "text": m.text,
                        "error": str(e),
                    }
                )
        structured["region_crops"] = region_crops
        structured["crop_output_dir"] = str(crop_dir)

    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=f"Detected {len(matches)} text region(s) in {path}",
            )
        ],
        structuredContent=structured,
        isError=False,
    )


def handle_crop_for_inspection(
    path: str,
    bbox: tuple[float, float, float, float],
    *,
    output_path: str | None = None,
    normalized: bool = False,
    padding: float = 0.0,
) -> CallToolResult:
    """Handle crop_for_inspection tool call.

    Args:
        path: Absolute path to source image.
        bbox: [x1, y1, x2, y2] region to crop.
        output_path: Where to save the crop (auto-generated if None).
        normalized: If True, bbox is in [0, 1] coords.
        padding: Fractional padding to add around bbox (0.1 = 10% on each side).

    Returns:
        CallToolResult with structuredContent containing crop info.
    """
    if not Path(path).is_file():
        return _error_result(f"File not found: {path}")

    try:
        result: CropResult = crop_region(
            path,
            bbox,
            output_path=output_path,
            normalized=normalized,
            padding=padding,
        )
    except CropError as e:
        return _error_result(str(e))

    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=f"Cropped region saved to {result.output_path}",
            )
        ],
        structuredContent={
            "image_path": path,
            "crop": result.to_dict(),
        },
        isError=False,
    )
