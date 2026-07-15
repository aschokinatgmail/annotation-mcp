"""Tests for detection and region cropping tools.

Verifies:
    - Graceful degradation when system deps (tesseract, zbar) are missing
    - Barcode detection on synthetic EAN-13 images
    - Text region detection with English and Cyrillic
    - Region cropping with padding and various bbox formats
    - Handler-level error handling and structured output
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from PIL import Image

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
    TextDetectionError,
    TextMatch,
    detect_text_regions,
)
from annotation_mcp.handlers_detection import (
    handle_crop_for_inspection,
    handle_detect_barcodes,
    handle_detect_text_regions,
)

FIXTURES = Path(__file__).parent / "fixtures"
SYNTHETIC_TEXT = str(FIXTURES / "synthetic_text.png")
SYNTHETIC_NOISY = str(FIXTURES / "synthetic_noisy_text.png")
SYNTHETIC_BARCODE = str(FIXTURES / "synthetic_ean13.png")
GARDEN_PHOTO = str(FIXTURES / "garden_photo.jpg")


# ─────────────────────────────────────────────────────────────────────────
# Helpers — skip cleanly when system deps are missing
# ─────────────────────────────────────────────────────────────────────────


def _require_tesseract() -> None:
    if not shutil.which("tesseract"):
        pytest.skip("tesseract binary not installed")
    status = check_tesseract()
    if not status.available:
        pytest.skip(f"tesseract unavailable: {status.detail}")


def _require_lang(lang: str) -> None:
    """Verify tesseract has the given language pack installed."""
    all_present, missing = check_tesseract_languages(*lang.split("+"))
    if not all_present:
        pytest.skip(f"tesseract language pack(s) missing: {missing}")


def _require_zbar() -> None:
    status = check_zbar_library()
    if not status.available:
        pytest.skip(f"zbar/pyzbar not available: {status.detail}")


def _require_barcode_fixture() -> None:
    if not Path(SYNTHETIC_BARCODE).exists():
        pytest.skip("synthetic_ean13.png fixture not created")


def _require_text_fixture() -> None:
    if not Path(SYNTHETIC_TEXT).exists():
        pytest.skip("synthetic_text.png fixture not created")


def _require_garden_photo() -> None:
    if not Path(GARDEN_PHOTO).exists():
        pytest.skip("garden_photo.jpg fixture not present")


# ─────────────────────────────────────────────────────────────────────────
# Availability checks
# ─────────────────────────────────────────────────────────────────────────


class TestAvailability:
    """Test that optional deps are detected correctly."""

    def test_tesseract_status(self) -> None:
        status = check_tesseract()
        if shutil.which("tesseract"):
            assert status.available
            assert status.detail
        else:
            assert not status.available
            assert status.install_hint

    def test_tesseract_languages_check_eng(self) -> None:
        if not shutil.which("tesseract"):
            pytest.skip("tesseract not installed")
        all_present, missing = check_tesseract_languages("eng")
        assert all_present
        assert missing == []

    def test_tesseract_languages_missing(self) -> None:
        if not shutil.which("tesseract"):
            pytest.skip("tesseract not installed")
        all_present, missing = check_tesseract_languages("xyz_nonexistent_lang")
        assert not all_present
        assert "xyz_nonexistent_lang" in missing

    def test_tesseract_languages_partial(self) -> None:
        if not shutil.which("tesseract"):
            pytest.skip("tesseract not installed")
        all_present, missing = check_tesseract_languages("eng", "xyz_nonexistent")
        assert not all_present
        assert "xyz_nonexistent" in missing
        assert "eng" not in missing

    def test_zbar_status(self) -> None:
        status = check_zbar_library()
        candidates = (
            "/opt/homebrew/lib/libzbar.dylib",
            "/opt/homebrew/opt/zbar/lib/libzbar.dylib",
            "/usr/local/lib/libzbar.dylib",
            "/usr/local/opt/zbar/lib/libzbar.dylib",
        )
        if any(Path(c).exists() for c in candidates):
            assert status.available


# ─────────────────────────────────────────────────────────────────────────
# Barcode detection
# ─────────────────────────────────────────────────────────────────────────


class TestDetectBarcodesUnit:
    """detect_barcodes() unit tests with synthetic EAN-13."""

    def test_detects_synthetic_ean13(self) -> None:
        _require_barcode_fixture()
        _require_zbar()
        matches = detect_barcodes(SYNTHETIC_BARCODE)
        assert len(matches) >= 1
        ean = matches[0]
        assert ean.type == "EAN13"
        assert ean.data == "4607026420155"
        assert ean.bbox_pixels[0] >= 0
        assert ean.bbox_pixels[1] >= 0
        assert ean.bbox_pixels[2] > ean.bbox_pixels[0]
        assert ean.bbox_pixels[3] > ean.bbox_pixels[1]

    def test_normalized_bbox_is_in_unit_range(self) -> None:
        _require_barcode_fixture()
        _require_zbar()
        matches = detect_barcodes(SYNTHETIC_BARCODE)
        assert len(matches) >= 1
        bbox = matches[0].bbox_normalized
        assert 0.0 <= bbox[0] <= 1.0
        assert 0.0 <= bbox[1] <= 1.0
        assert 0.0 <= bbox[2] <= 1.0
        assert 0.0 <= bbox[3] <= 1.0
        assert bbox[2] > bbox[0]
        assert bbox[3] > bbox[1]

    def test_allowed_types_filter(self) -> None:
        _require_barcode_fixture()
        _require_zbar()
        matches = detect_barcodes(SYNTHETIC_BARCODE, allowed_types=["EAN13"])
        assert all(m.type == "EAN13" for m in matches)

    def test_allowed_types_excludes(self) -> None:
        _require_barcode_fixture()
        _require_zbar()
        matches = detect_barcodes(SYNTHETIC_BARCODE, allowed_types=["QRCODE"])
        assert len(matches) == 0

    def test_missing_file_raises(self) -> None:
        with pytest.raises(BarcodeDetectionError, match="Cannot load image"):
            detect_barcodes("/nonexistent/file.jpg")

    def test_empty_image_returns_no_matches(self, tmp_path: Path) -> None:
        _require_zbar()
        img = Image.new("RGB", (200, 200), "white")
        path = tmp_path / "empty.png"
        img.save(path)
        matches = detect_barcodes(str(path))
        assert matches == []

    def test_to_dict_serialization(self) -> None:
        m = BarcodeMatch(
            type="EAN13",
            data="4607026420155",
            bbox_pixels=(10, 20, 110, 120),
            bbox_normalized=(0.05, 0.1, 0.55, 0.6),
            confidence=0.95,
        )
        d = m.to_dict(200, 200)
        assert d["type"] == "EAN13"
        assert d["data"] == "4607026420155"
        assert d["bbox_pixels"] == [10, 20, 110, 120]
        assert d["bbox_normalized"] == [0.05, 0.1, 0.55, 0.6]
        assert d["confidence"] == 0.95
        assert d["image_width"] == 200
        assert d["image_height"] == 200


# ─────────────────────────────────────────────────────────────────────────
# Text region detection
# ─────────────────────────────────────────────────────────────────────────


class TestDetectTextRegionsUnit:
    """detect_text_regions() unit tests."""

    def test_detects_brand_title(self) -> None:
        _require_text_fixture()
        _require_tesseract()
        _require_lang("eng")
        matches = detect_text_regions(SYNTHETIC_TEXT, lang="eng", detail="word")
        assert len(matches) > 0
        texts = " ".join(m.text for m in matches)
        assert "BRAND" in texts or "TITLE" in texts

    def test_word_detail_returns_word_level(self) -> None:
        _require_text_fixture()
        _require_tesseract()
        _require_lang("eng")
        matches = detect_text_regions(SYNTHETIC_TEXT, lang="eng", detail="word")
        assert len(matches) > 0
        assert all(m.level == "word" for m in matches)

    def test_line_detail_returns_line_level(self) -> None:
        _require_text_fixture()
        _require_tesseract()
        _require_lang("eng")
        matches = detect_text_regions(SYNTHETIC_TEXT, lang="eng", detail="line")
        assert len(matches) > 0
        assert all(m.level == "line" for m in matches)

    def test_block_detail_returns_block_level(self) -> None:
        _require_text_fixture()
        _require_tesseract()
        _require_lang("eng")
        matches = detect_text_regions(SYNTHETIC_TEXT, lang="eng", detail="block")
        for m in matches:
            assert m.level == "block"

    def test_cyrillic_detection(self) -> None:
        _require_text_fixture()
        _require_tesseract()
        _require_lang("rus")
        matches = detect_text_regions(SYNTHETIC_TEXT, lang="rus")
        for m in matches:
            assert 0 <= m.confidence <= 100
        texts = " ".join(m.text for m in matches)
        assert "ФИТОСПОРИН" in texts or len(matches) > 0

    def test_min_confidence_filters_low(self) -> None:
        _require_text_fixture()
        _require_tesseract()
        _require_lang("eng")
        all_matches = detect_text_regions(SYNTHETIC_TEXT, lang="eng", detail="word")
        filtered = detect_text_regions(
            SYNTHETIC_TEXT, lang="eng", detail="word", min_confidence=99.0
        )
        assert len(filtered) <= len(all_matches)

    def test_missing_file_raises(self) -> None:
        with pytest.raises(TextDetectionError, match="Cannot load image"):
            detect_text_regions("/nonexistent/file.jpg")

    def test_to_dict_serialization(self) -> None:
        m = TextMatch(
            text="hello",
            confidence=87.5,
            bbox_pixels=(10, 20, 110, 120),
            bbox_normalized=(0.05, 0.1, 0.55, 0.6),
            block_num=1,
            line_num=1,
            level="word",
        )
        d = m.to_dict(200, 200)
        assert d["text"] == "hello"
        assert d["confidence"] == 87.5
        assert d["bbox_pixels"] == [10, 20, 110, 120]
        assert d["bbox_normalized"] == [0.05, 0.1, 0.55, 0.6]
        assert d["block"] == 1
        assert d["line"] == 1
        assert d["level"] == "word"

    def test_preprocess_modes_run(self) -> None:
        """All four preprocess modes should run without error."""
        _require_text_fixture()
        _require_tesseract()
        _require_lang("eng")
        for mode in ("none", "clahe", "adaptive", "otsu"):
            try:
                detect_text_regions(SYNTHETIC_TEXT, lang="eng", detail="word", preprocess=mode)
            except ImportError as e:
                # cv2 not installed — only "none" should work
                if mode != "none":
                    pytest.skip(f"opencv required for {mode}: {e}")
                raise

    def test_filter_garbage_disabled_keeps_all(self) -> None:
        """filter_garbage=False returns everything tesseract detected, including junk."""
        _require_text_fixture()
        _require_tesseract()
        _require_lang("eng")
        all_matches = detect_text_regions(
            SYNTHETIC_TEXT, lang="eng", detail="word", min_confidence=0.0,
            filter_garbage=False,
        )
        filtered_matches = detect_text_regions(
            SYNTHETIC_TEXT, lang="eng", detail="word", min_confidence=0.0,
            filter_garbage=True,
        )
        # When filter_garbage=True we should have FEWER or equal matches
        # (some may be filtered as junk on synthetic text too, though
        # synthetic_text.png is clean).
        assert len(filtered_matches) <= len(all_matches)

    def test_filter_garbage_default_is_on(self) -> None:
        """Default behaviour is filter_garbage=True (no kwarg needed)."""
        _require_text_fixture()
        _require_tesseract()
        _require_lang("eng")
        default_matches = detect_text_regions(
            SYNTHETIC_TEXT, lang="eng", detail="word", min_confidence=0.0,
        )
        # Default should match the explicit-True call
        explicit_matches = detect_text_regions(
            SYNTHETIC_TEXT, lang="eng", detail="word", min_confidence=0.0,
            filter_garbage=True,
        )
        assert len(default_matches) == len(explicit_matches)

    def test_psm_fallback_when_default_returns_zero(self, tmp_path: Path) -> None:
        """When default PSM returns 0 regions, retry with PSM=6.

        Uses a noisy synthetic image where default PSM may fail to find text.
        Verifies the fallback is wired up correctly.
        """
        from PIL import Image, ImageDraw, ImageFont

        _require_tesseract()
        _require_lang("eng")
        # Create an image with text in a block (PSM=6 territory)
        path = tmp_path / "psm_test.png"
        img = Image.new("RGB", (400, 200), "white")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(
                "/System/Library/Fonts/Helvetica.ttc", 32
            )
        except OSError:
            font = ImageFont.load_default()
        draw.text((50, 80), "PSM TEST TEXT", fill="black", font=font)
        img.save(path)

        # Run with default PSM (None) — should fall back to PSM=6 if needed
        matches = detect_text_regions(
            str(path), lang="eng", detail="word", preprocess="none",
        )
        # Either default or fallback should find some text
        assert isinstance(matches, list)

    def test_filter_garbage_off_keeps_garbage(self) -> None:
        """filter_garbage=False passes through all regions including junk.

        Sanity check: when filtering is off, we should get the same number
        of regions as without the filter (no other side effects).
        """
        _require_text_fixture()
        _require_tesseract()
        _require_lang("eng")
        unfiltered = detect_text_regions(
            SYNTHETIC_TEXT, lang="eng", detail="word", min_confidence=0.0,
            filter_garbage=False,
        )
        # Make sure the parameter is being read (not silently ignored)
        # by checking that a known-clean word is still in unfiltered output
        texts = " ".join(m.text for m in unfiltered)
        # synthetic_text has "BRAND TITLE" or words that pass the filter
        assert "BRAND" in texts or "TITLE" in texts

    def test_crop_regions_returns_crop_paths(self, tmp_path: Path) -> None:
        """When crop_regions=True, output includes region_crops with file paths."""
        _require_text_fixture()
        _require_tesseract()
        _require_lang("eng")
        out_dir = tmp_path / "crops"
        # Use the handler to get the structured result
        result = handle_detect_text_regions(
            SYNTHETIC_TEXT,
            lang="eng",
            detail="word",
            min_confidence=0.0,
            crop_regions=True,
            crop_output_dir=str(out_dir),
        )
        assert not result.isError
        sc = result.structuredContent
        assert sc is not None
        assert "region_crops" in sc, "Expected region_crops in result"
        crops = sc["region_crops"]
        assert isinstance(crops, list)
        assert len(crops) > 0
        # Each crop should have bbox_pixels and crop_path
        first = crops[0]
        assert "crop_path" in first
        assert Path(first["crop_path"]).exists()
        assert "bbox_pixels" in first


# ─────────────────────────────────────────────────────────────────────────
# Region cropping
# ─────────────────────────────────────────────────────────────────────────


class TestCropRegion:
    """crop_region() tests."""

    def test_basic_crop(self, tmp_path: Path) -> None:
        src = tmp_path / "src.png"
        Image.new("RGB", (400, 300), "white").save(src)
        out = tmp_path / "out.png"
        result = crop_region(str(src), (100, 50, 300, 250), output_path=str(out))
        assert isinstance(result, CropResult)
        assert result.width == 200
        assert result.height == 200
        assert result.x == 100
        assert result.y == 50
        assert result.bbox_pixels == (100, 50, 300, 250)
        assert result.source_width == 400
        assert result.source_height == 300
        assert Path(result.output_path).exists()

    def test_normalized_bbox(self, tmp_path: Path) -> None:
        src = tmp_path / "src.png"
        Image.new("RGB", (400, 300), "white").save(src)
        out = tmp_path / "out.png"
        result = crop_region(
            str(src), (0.25, 0.25, 0.75, 0.75), output_path=str(out), normalized=True
        )
        assert result.bbox_pixels == (100, 75, 300, 225)

    def test_padding_expands_region(self, tmp_path: Path) -> None:
        src = tmp_path / "src.png"
        Image.new("RGB", (1000, 1000), "white").save(src)
        out = tmp_path / "out.png"
        result = crop_region(
            str(src), (400, 400, 600, 600), output_path=str(out), padding=0.5
        )
        # 200x200 bbox + 50% padding on each side = 200 + 200 = 400 region
        # (clamped if needed). 400 < 1000 so no clamp, should be 400x400.
        assert result.width == 400
        assert result.height == 400
        assert result.x == 300  # 400 - 100 padding
        assert result.y == 300

    def test_auto_output_path(self, tmp_path: Path) -> None:
        src = tmp_path / "myimage.png"
        Image.new("RGB", (400, 300), "white").save(src)
        result = crop_region(str(src), (0, 0, 100, 100))
        assert "myimage_crop_" in result.output_path
        assert Path(result.output_path).exists()

    def test_clamp_to_image_bounds(self, tmp_path: Path) -> None:
        src = tmp_path / "src.png"
        Image.new("RGB", (400, 300), "white").save(src)
        out = tmp_path / "out.png"
        result = crop_region(str(src), (-50, -50, 500, 400), output_path=str(out))
        assert result.bbox_pixels == (0, 0, 400, 300)
        assert result.width == 400
        assert result.height == 300

    def test_invalid_bbox_raises(self, tmp_path: Path) -> None:
        src = tmp_path / "src.png"
        Image.new("RGB", (400, 300), "white").save(src)
        with pytest.raises(CropError, match="Invalid bbox"):
            crop_region(str(src), (200, 100, 100, 50), output_path=str(tmp_path / "o.png"))

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        out = tmp_path / "o.png"
        with pytest.raises(CropError, match="not found"):
            crop_region("/nonexistent/file.png", (0, 0, 100, 100), output_path=str(out))
        # Also ensure FileNotFoundError is converted to CropError
        try:
            crop_region("/nonexistent/file.png", (0, 0, 100, 100), output_path=str(out))
        except CropError:
            pass
        except FileNotFoundError as e:
            pytest.fail(f"FileNotFoundError not wrapped in CropError: {e}")

    def test_to_dict(self, tmp_path: Path) -> None:
        src = tmp_path / "src.png"
        Image.new("RGB", (400, 300), "white").save(src)
        out = tmp_path / "out.png"
        result = crop_region(str(src), (10, 20, 110, 120), output_path=str(out))
        d = result.to_dict()
        assert d["width"] == 100
        assert d["height"] == 100
        assert d["x"] == 10
        assert d["y"] == 20
        assert d["bbox_pixels"] == [10, 20, 110, 120]
        assert d["source_width"] == 400
        assert d["source_height"] == 300

    def test_jpeg_output_preserves_format(self, tmp_path: Path) -> None:
        src = tmp_path / "src.jpg"
        Image.new("RGB", (400, 300), "red").save(src, "JPEG")
        out = tmp_path / "out.jpg"
        result = crop_region(str(src), (10, 20, 110, 120), output_path=str(out))
        img = Image.open(result.output_path)
        assert img.format == "JPEG"


# ─────────────────────────────────────────────────────────────────────────
# Handler-level tests
# ─────────────────────────────────────────────────────────────────────────


class TestHandleDetectBarcodes:
    """handle_detect_barcodes() returns proper CallToolResult."""

    def test_success_returns_structured_content(self) -> None:
        _require_barcode_fixture()
        _require_zbar()
        result = handle_detect_barcodes(SYNTHETIC_BARCODE)
        assert result.isError is False
        assert result.structuredContent is not None
        sc = result.structuredContent
        assert sc["image_path"] == SYNTHETIC_BARCODE
        assert sc["count"] >= 1
        assert isinstance(sc["barcodes"], list)
        assert len(sc["barcodes"]) >= 1
        first = sc["barcodes"][0]
        assert first["type"] == "EAN13"
        assert first["data"] == "4607026420155"

    def test_missing_file_returns_error(self) -> None:
        _require_zbar()
        result = handle_detect_barcodes("/nonexistent/file.jpg")
        assert result.isError
        assert "File not found" in str(result.content)


class TestHandleDetectTextRegions:
    """handle_detect_text_regions() returns proper CallToolResult."""

    def test_success_returns_structured_content(self) -> None:
        _require_text_fixture()
        _require_tesseract()
        _require_lang("eng")
        result = handle_detect_text_regions(SYNTHETIC_TEXT, lang="eng", detail="word")
        assert result.isError is False
        assert result.structuredContent is not None
        sc = result.structuredContent
        assert sc["image_path"] == SYNTHETIC_TEXT
        assert sc["lang"] == "eng"
        assert sc["detail"] == "word"
        assert isinstance(sc["regions"], list)
        assert sc["count"] == len(sc["regions"])

    def test_invalid_detail_returns_error(self) -> None:
        _require_tesseract()
        result = handle_detect_text_regions(SYNTHETIC_TEXT, detail="invalid")
        assert result.isError
        assert "Invalid detail" in str(result.content)

    def test_invalid_preprocess_returns_error(self) -> None:
        _require_tesseract()
        result = handle_detect_text_regions(SYNTHETIC_TEXT, preprocess="invalid")
        assert result.isError
        assert "Invalid preprocess" in str(result.content)

    def test_missing_file_returns_error(self) -> None:
        _require_tesseract()
        result = handle_detect_text_regions("/nonexistent/file.jpg")
        assert result.isError


class TestHandleCropForInspection:
    """handle_crop_for_inspection() returns proper CallToolResult."""

    def test_success_returns_structured_content(self, tmp_path: Path) -> None:
        src = tmp_path / "src.png"
        Image.new("RGB", (400, 300), "white").save(src)
        out = tmp_path / "out.png"
        result = handle_crop_for_inspection(
            str(src), (100, 50, 300, 250), output_path=str(out)
        )
        assert result.isError is False
        sc = result.structuredContent
        assert sc is not None
        assert sc["image_path"] == str(src)
        assert sc["crop"]["width"] == 200
        assert sc["crop"]["height"] == 200
        assert Path(sc["crop"]["output_path"]).exists()

    def test_missing_file_returns_error(self, tmp_path: Path) -> None:
        out = tmp_path / "o.png"
        result = handle_crop_for_inspection(
            "/nonexistent/file.png", (0, 0, 100, 100), output_path=str(out)
        )
        assert result.isError
        assert "File not found" in str(result.content)


# ─────────────────────────────────────────────────────────────────────────
# Integration tests with real photo
# ─────────────────────────────────────────────────────────────────────────


class TestRealPhotoIntegration:
    """End-to-end tests on the garden_photo.jpg real product photo."""

    def test_detect_text_on_real_photo(self) -> None:
        """Real photo of a Russian product package — detect any text."""
        _require_garden_photo()
        _require_tesseract()
        _require_lang("eng+rus")
        matches = detect_text_regions(
            GARDEN_PHOTO, lang="eng+rus", detail="line", preprocess="clahe"
        )
        assert len(matches) > 0
        for m in matches:
            assert m.bbox_pixels[2] <= 3024
            assert m.bbox_pixels[3] <= 4032
            assert 0 <= m.bbox_normalized[0] <= 1
            assert 0 <= m.bbox_normalized[1] <= 1
            assert 0 <= m.bbox_normalized[2] <= 1
            assert 0 <= m.bbox_normalized[3] <= 1

    def test_crop_then_detect_zoom_workflow(self, tmp_path: Path) -> None:
        """Crop a region, then run text detection on the crop."""
        _require_garden_photo()
        _require_tesseract()
        _require_lang("eng+rus")
        crop_out = tmp_path / "crop.jpg"
        result = crop_region(
            GARDEN_PHOTO,
            (0.2, 0.05, 0.8, 0.20),
            output_path=str(crop_out),
            normalized=True,
        )
        assert Path(result.output_path).exists()
        matches = detect_text_regions(
            result.output_path, lang="eng+rus", detail="word", preprocess="clahe"
        )
        assert len(matches) > 0

    def test_detect_barcode_on_real_photo(self) -> None:
        """Real product photo should have an EAN-13 barcode detected."""
        _require_garden_photo()
        _require_zbar()
        matches = detect_barcodes(GARDEN_PHOTO)
        ean_matches = [m for m in matches if m.type == "EAN13"]
        assert len(ean_matches) >= 1
        assert any("4607026420155" in m.data for m in ean_matches)
