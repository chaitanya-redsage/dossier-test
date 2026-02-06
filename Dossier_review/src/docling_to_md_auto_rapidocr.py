#!/usr/bin/env python3
"""
Minimal helper for file_extract_router.py.
Exposes convert_to_markdown() using Docling with OCR enabled.
"""

from __future__ import annotations

import sys
from pathlib import Path


def convert_to_markdown(input_path: str | Path) -> str:
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")

    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            OcrAutoOptions,
            OcrMacOptions,
            RapidOcrOptions,
            ThreadedPdfPipelineOptions,
        )
        from docling.document_converter import (
            DocumentConverter,
            ImageFormatOption,
            PdfFormatOption,
        )
    except ImportError as exc:
        raise SystemExit(
            "Docling is not installed. Install it with:\n"
            "  python -m pip install docling"
        ) from exc

    if sys.platform == "darwin":
        ocr_options = OcrMacOptions(
            lang=["en-US"],
            force_full_page_ocr=True,
            bitmap_area_threshold=0.0,
        )
    else:
        try:
            ocr_options = RapidOcrOptions(
                lang=["english"],
                force_full_page_ocr=True,
                bitmap_area_threshold=0.0,
            )
        except Exception:
            ocr_options = OcrAutoOptions(
                lang=[],
                force_full_page_ocr=True,
                bitmap_area_threshold=0.0,
            )

    pipeline_options = ThreadedPdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True,
        ocr_options=ocr_options,
    )

    format_options = {
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options),
    }

    converter = DocumentConverter(format_options=format_options)
    result = converter.convert(str(p))
    return result.document.export_to_markdown()


__all__ = ["convert_to_markdown"]
