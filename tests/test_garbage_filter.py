"""Tests for the OCR garbage-text filter.

The filter rejects OCR noise that Tesseract produces on real-world photos
(high-confidence nonsense like 'VLOAOL', 'yOLO', 'aS', '||||').

These tests cover the false-positive (real text wrongly rejected) and
false-negative (garbage wrongly accepted) boundaries.
"""
from __future__ import annotations

from annotation_mcp.detection.garbage_filter import (
    _char_class_diversity,
    _has_real_token,
    _looks_like_part_number,
    _vowel_ratio,
    looks_like_garbage,
)


class TestCharClassDiversity:
    """_char_class_diversity: distinct chars / total length."""

    def test_all_distinct(self) -> None:
        assert _char_class_diversity("abcd") == 1.0

    def test_all_same(self) -> None:
        assert _char_class_diversity("aaaa") == 0.25

    def test_empty(self) -> None:
        assert _char_class_diversity("") == 0.0

    def test_half_distinct(self) -> None:
        assert _char_class_diversity("abab") == 0.5


class TestVowelRatio:
    """_vowel_ratio: fraction of Latin alpha chars that are vowels."""

    def test_all_vowels(self) -> None:
        assert _vowel_ratio("aeiou") == 1.0

    def test_no_vowels(self) -> None:
        # "rhythm" has 'y' which counts as a vowel here (1/6)
        assert abs(_vowel_ratio("rhythm") - 1/6) < 0.01
        # "bcdfgh" has no vowels at all
        assert _vowel_ratio("bcdfgh") == 0.0

    def test_mixed(self) -> None:
        # "toyota" has o, o, a, y as vowels (4/6) — y counts as a vowel
        assert abs(_vowel_ratio("toyota") - 4/6) < 0.01

    def test_no_ascii_alpha(self) -> None:
        # Cyrillic/CJK — return -1 (exempt from check)
        assert _vowel_ratio("ФИТО") == -1.0
        assert _vowel_ratio("引擎") == -1.0

    def test_with_digits(self) -> None:
        # "12V" — 1 alpha (V), 0 vowels. ratio = 0.0
        assert _vowel_ratio("12V") == 0.0


class TestPartNumber:
    """_looks_like_part_number: matches real-world part-number patterns."""

    def test_mixed_letters_digits_long(self) -> None:
        assert _looks_like_part_number("27060-54170")
        assert _looks_like_part_number("4607026420155")
        assert _looks_like_part_number("CODE128")
        assert _looks_like_part_number("QR-1")
        # ALT-S is pure-letters, not a part-number pattern
        assert not _looks_like_part_number("ALT-S")

    def test_too_short_mixed(self) -> None:
        # Even mixed letter/digit fails if too short (<3 chars)
        assert not _looks_like_part_number("12")

    def test_pure_letters(self) -> None:
        # All-letters with no digits is not a part-number pattern
        assert not _looks_like_part_number("TOYOTA")
        assert not _looks_like_part_number("BRAKE")

    def test_pure_digits(self) -> None:
        # All-digits: needs ≥4 chars
        assert not _looks_like_part_number("04")
        assert not _looks_like_part_number("000")
        assert _looks_like_part_number("0000")
        assert _looks_like_part_number("2204008")
        assert _looks_like_part_number("12345678")

    def test_invalid_chars(self) -> None:
        # Spaces, slashes, etc. are not part-number patterns
        assert not _looks_like_part_number("12V 65A")
        assert not _looks_like_part_number("ALT/S")


class TestHasRealToken:
    """_has_real_token: any plausible real text token."""

    def test_english_word(self) -> None:
        assert _has_real_token("TOYOTA")
        assert _has_real_token("CAUTION")
        assert _has_real_token("BRAKE")
        assert _has_real_token("INSPECT")

    def test_brand_name(self) -> None:
        assert _has_real_token("DENSO")
        assert _has_real_token("toyota")

    def test_part_number(self) -> None:
        assert _has_real_token("27060-54170")
        assert _has_real_token("4607026420155")

    def test_cyrillic_or_cjk(self) -> None:
        assert _has_real_token("ФИТОСПОРИН")
        assert _has_real_token("发动机")
        assert _has_real_token("ТОЙОТА")

    def test_pure_garbage(self) -> None:
        assert not _has_real_token("VLOAOL")
        assert not _has_real_token("yOLO")
        assert not _has_real_token("Wap")
        assert not _has_real_token("04")
        assert not _has_real_token("VLLL")

    def test_mixed_garbage_with_real(self) -> None:
        # "yOLO TOYOTA" should pass (has real token)
        assert _has_real_token("yOLO TOYOTA")
        # "VLOAOL" alone fails
        assert not _has_real_token("VLOAOL")


class TestLooksLikeGarbage:
    """End-to-end looks_like_garbage() checks."""

    def test_real_text_passes(self) -> None:
        assert not looks_like_garbage("TOYOTA")
        assert not looks_like_garbage("ФИТОСПОРИН")
        assert not looks_like_garbage("TOYOTA MOTOR CORPORATION")
        assert not looks_like_garbage("DENSO")
        assert not looks_like_garbage("CAUTION")
        assert not looks_like_garbage("POWER STEERING FLUID")
        assert not looks_like_garbage("BRAKE")
        assert not looks_like_garbage("DANGER")
        assert not looks_like_garbage("ENGINE")
        assert not looks_like_garbage("WARNING")
        assert not looks_like_garbage("USE ONLY DOT 3")
        assert not looks_like_garbage("CAUTION DO NOT TOUCH")
        assert not looks_like_garbage("27060-54170")
        assert not looks_like_garbage("4607026420155")
        assert not looks_like_garbage("CODE128")
        assert not looks_like_garbage("MADE IN JAPAN")

    def test_garbage_is_rejected(self) -> None:
        assert looks_like_garbage("VLOAOL")
        assert looks_like_garbage("yOLO")
        assert looks_like_garbage("Wap")
        assert looks_like_garbage("shy")
        assert looks_like_garbage("VLO")
        assert looks_like_garbage("VLOA")
        assert looks_like_garbage("SPAS")
        assert looks_like_garbage("tee")
        assert looks_like_garbage("hoa")
        assert looks_like_garbage("lym")
        assert looks_like_garbage("yayy")
        assert looks_like_garbage("aaaa")
        assert looks_like_garbage("||||")
        assert looks_like_garbage("hhhh")
        assert looks_like_garbage("SS")
        assert looks_like_garbage("VLLL")
        assert looks_like_garbage("KAT")
        assert looks_like_garbage("MRE")
        assert looks_like_garbage("mo!")
        assert looks_like_garbage("aS")

    def test_short_strings_rejected(self) -> None:
        assert looks_like_garbage("a")
        assert looks_like_garbage("|")
        assert looks_like_garbage("2")
        assert looks_like_garbage("")

    def test_punctuation_only_rejected(self) -> None:
        assert looks_like_garbage("|||")
        assert looks_like_garbage("---")
        assert looks_like_garbage("...")

    def test_actual_project_garbage(self) -> None:
        """Reproduce the actual garbage strings observed in real-world OCR tests."""
        for junk in [
            "VLOAOL", "yOLO", "Wap", "shy", "pH", "Sat", "tS", "ok", "S",
            "ay", "we", "hoa", "lym", "yay", "KAT", "MRE", "tee",
            "VLLL", "VLO", "VLOA", "SPAS", "O!", "Kat", "aB", "ad",
        ]:
            assert looks_like_garbage(junk), f"Should be garbage: {junk!r}"

    def test_actual_project_real(self) -> None:
        """Reproduce the actual real strings observed in real-world OCR tests."""
        for real in [
            "TOYOTA", "DENSO", "CAUTION", "BRAKE", "ENGINE", "WARNING",
            "INSPECT", "CORPORATION", "MADE IN JAPAN", "POWER STEERING",
        ]:
            assert not looks_like_garbage(real), f"Should pass: {real!r}"
