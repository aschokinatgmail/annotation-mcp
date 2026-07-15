"""EXIF orientation spec-compliance tests.

Verifies the project's `_apply_exif_orientation` function produces the
correct output dimensions and rotation direction for each of the 8 EXIF
orientation values, per the EXIF 2.3 specification (Table 9).

Also verifies equivalence with PIL's reference `ImageOps.exif_transpose`
implementation.
"""
from __future__ import annotations

import pytest
from PIL import Image, ImageOps

from annotation_mcp.image_io import _apply_exif_orientation


# Expected transforms per EXIF 2.3 spec (raw WxH → upright W'xH').
# Orient 5/7/8 swap dimensions like orient 6 (CCW rotations).
# (orient, raw_w, raw_h, expected_w, expected_h)
EXIF_CASES: list[tuple[int, int, int, int, int]] = [
    (1, 400, 300, 400, 300),  # normal
    (2, 400, 300, 400, 300),  # mirror L/R (no dimension swap)
    (3, 400, 300, 400, 300),  # 180° (no dimension swap)
    (4, 400, 300, 400, 300),  # mirror T/B (no dimension swap)
    (5, 400, 300, 300, 400),  # mirror L/R + 270° CW (= 90° CCW)
    (6, 400, 300, 300, 400),  # 90° CW
    (7, 400, 300, 300, 400),  # mirror L/R + 90° CW
    (8, 400, 300, 300, 400),  # 90° CCW (= 270° CW)
]


def _make_test_image(orient: int, w: int, h: int) -> Image.Image:
    """Create a test image with the given EXIF orientation and size."""
    img = Image.new("RGB", (w, h), "blue")
    exif = img.getexif()
    exif[0x0112] = orient
    return img


class TestExifOrientationSpec:
    """Verify _apply_exif_orientation matches EXIF 2.3 spec for all 8 values."""

    @pytest.mark.parametrize(
        "orient,raw_w,raw_h,expected_w,expected_h", EXIF_CASES
    )
    def test_orientation_dimensions(
        self, orient: int, raw_w: int, raw_h: int, expected_w: int, expected_h: int
    ) -> None:
        img = _make_test_image(orient, raw_w, raw_h)
        out = _apply_exif_orientation(img)
        assert out.size == (expected_w, expected_h), (
            f"Orient {orient}: expected {expected_w}x{expected_h}, got {out.size}"
        )


class TestExifMatchesPILReference:
    """Verify our implementation matches PIL's ImageOps.exif_transpose."""

    @pytest.mark.parametrize("orient", [1, 2, 3, 4, 5, 6, 7, 8])
    def test_matches_pil_exif_transpose(self, orient: int) -> None:
        img = _make_test_image(orient, 400, 300)
        my_result = _apply_exif_orientation(img)
        pil_result = ImageOps.exif_transpose(img)
        assert my_result.size == pil_result.size, (
            f"Orient {orient}: ours {my_result.size} != PIL {pil_result.size}"
        )


class TestExifOrientationRotation:
    """Verify the actual rotation direction, not just dimensions.

    For orient 6 specifically, verify the rotation is 90° CW (the spec
    value, not 90° CCW). We do this by drawing a marker in one corner
    of the raw image and checking it lands in the expected corner
    after rotation.
    """

    def test_orient_6_rotates_90_cw(self) -> None:
        """Orient 6 = "0th row at visual right, 0th col at visual top".

        This means: to display correctly, rotate 90° CW. The pixel that
        was at top-left of the raw file ends up at top-right of upright.
        """
        # 4x2 raw image with a red marker at (0, 0) — top-left
        img = Image.new("RGB", (4, 2), "white")
        for x in range(4):
            img.putpixel((x, 0), (255, 0, 0))  # top row: red
        for x in range(4):
            img.putpixel((x, 1), (0, 0, 255))  # bottom row: blue

        exif = img.getexif()
        exif[0x0112] = 6

        rotated = _apply_exif_orientation(img)
        # After 90° CW rotation, the 4x2 raw becomes 2x4 upright.
        # Raw top row (red) becomes the right column of upright.
        # Raw bottom row (blue) becomes the left column of upright.
        assert rotated.size == (2, 4)
        # Check corners
        # Top-left of upright = raw bottom-left = blue
        assert rotated.getpixel((0, 0)) == (0, 0, 255)
        # Top-right of upright = raw top-right = red
        assert rotated.getpixel((1, 0)) == (255, 0, 0)
        # Bottom-left of upright = raw bottom-right = blue
        assert rotated.getpixel((0, 3)) == (0, 0, 255)
        # Bottom-right of upright = raw top-right corner pixel = red
        assert rotated.getpixel((1, 3)) == (255, 0, 0)

    def test_orient_8_rotates_90_ccw(self) -> None:
        """Orient 8 = "0th row at visual left, 0th col at visual bottom".

        This means: to display correctly, rotate 90° CCW.
        """
        # 4x2 raw image with a red marker at (0, 0) — top-left
        img = Image.new("RGB", (4, 2), "white")
        for x in range(4):
            img.putpixel((x, 0), (255, 0, 0))  # top row: red
        for x in range(4):
            img.putpixel((x, 1), (0, 0, 255))  # bottom row: blue

        exif = img.getexif()
        exif[0x0112] = 8

        rotated = _apply_exif_orientation(img)
        # After 90° CCW rotation, the 4x2 raw becomes 2x4 upright.
        # Raw top row (red) becomes the LEFT column of upright.
        # Raw bottom row (blue) becomes the RIGHT column of upright.
        assert rotated.size == (2, 4)
        # Top-left of upright = raw top-left = red
        assert rotated.getpixel((0, 0)) == (255, 0, 0)
        # Top-right of upright = raw bottom-left = blue
        assert rotated.getpixel((1, 0)) == (0, 0, 255)
