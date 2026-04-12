# -*- coding: utf-8 -*-
"""
Resolve the Tesseract executable for pytesseract.

Order:
1) TESSERACT_CMD or TESSERACT_PATH — full path to tesseract.exe (recommended on Windows if not on PATH)
2) shutil.which("tesseract") — normal PATH lookup
3) Common Windows install locations (UB-Mannheim / Chocolatey default)
4) Fallback command name "tesseract" (pytesseract default)
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

_applied = False


def apply_pytesseract_cmd() -> str:
    """Configure pytesseract.pytesseract.tesseract_cmd; return the string in use."""
    global _applied
    import pytesseract

    if _applied:
        return str(pytesseract.pytesseract.tesseract_cmd)

    explicit = (os.getenv("TESSERACT_CMD") or os.getenv("TESSERACT_PATH") or "").strip()
    if explicit:
        ep = Path(explicit)
        if ep.is_file():
            pytesseract.pytesseract.tesseract_cmd = str(ep)
            _applied = True
            return str(ep)

    which = shutil.which("tesseract")
    if which:
        pytesseract.pytesseract.tesseract_cmd = which
        _applied = True
        return which

    for p in (
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
    ):
        if p.is_file():
            pytesseract.pytesseract.tesseract_cmd = str(p)
            _applied = True
            return str(p)

    pytesseract.pytesseract.tesseract_cmd = "tesseract"
    _applied = True
    return "tesseract"
