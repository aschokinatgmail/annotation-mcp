"""Object detection and OCR modules for coordinate acquisition.

These tools help the AI (or user) discover pixel coordinates of objects
in an image so that downstream annotation can be precise. They are
designed to fail gracefully when system dependencies (tesseract, zbar)
are not installed — the rest of the MCP server continues to work.

Modules:
    barcode: Pyzbar wrapper for 1D/2D barcode detection.
    ocr: Tesseract wrapper with optional preprocessing.
    crop: Region extraction helper for iterative refinement.
    availability: Runtime feature detection and dependency checks.
"""
