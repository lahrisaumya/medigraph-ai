"""
================================================================================
FILE:    backend/utils/pdf_extractor.py
PURPOSE: Healthcare PDF text extraction using PyMuPDF (fitz).
         Primary extraction method: native PDF text layer.
         Fallback for scanned pages: pytesseract OCR at 300 DPI.
         Called exclusively by Agent 1 (Document Analysis Agent).

PUBLIC API:
    extract_text_from_pdf(pdf_bytes)   → (text, page_count, is_scanned)
    extract_text_from_file(file_path)  → (text, page_count, is_scanned)
    clean_extracted_text(text)         → str
    get_pdf_metadata(pdf_bytes)        → dict
    _is_page_empty(text)               → bool   (also used in tests)

EXTRACTION STRATEGY per page:
    1. page.get_text("text")    — fast native extraction
    2. page.get_text("blocks")  — layout-aware block extraction (fallback)
    3. pytesseract OCR          — for scanned/image-only pages
    4. "[Scanned page…]"        — if tesseract not installed

QUALITY SIGNALS:
    is_scanned = True when >50% of pages required OCR
    Page text considered empty when <50 non-whitespace characters

DEPENDENCIES:
    pymupdf (fitz)    — pip install pymupdf
    pytesseract       — pip install pytesseract  (+ system tesseract binary)
    Pillow            — pip install Pillow        (image processing for OCR)

TESSERACT INSTALLATION:
    Ubuntu/Debian: sudo apt-get install tesseract-ocr
    macOS:         brew install tesseract
    Windows:       https://github.com/UB-Mannheim/tesseract/wiki
================================================================================
"""

import io
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
_MIN_CHARS_PER_PAGE  = 50    # pages with fewer non-whitespace chars → try OCR
_OCR_DPI             = 300   # dots per inch for OCR rasterisation
_OCR_PSM             = 6     # tesseract page segmentation mode (6=uniform block)
_MAX_TEXT_LENGTH     = 50_000  # truncate very long documents (LLM context limit)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN EXTRACTION — from bytes
# ──────────────────────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_bytes: bytes) -> Tuple[str, int, bool]:
    """
    Extract all text from a PDF given as raw bytes.
    This is the primary function called by Agent 1.

    Extraction per page (in priority order):
        1. Native text layer  (digital PDFs — instant)
        2. Block-based text   (tables, multi-column layouts)
        3. OCR via pytesseract (scanned / image pages)

    Args:
        pdf_bytes: Raw PDF file content as bytes.

    Returns:
        Tuple of:
            text       (str)  — Full extracted text from all pages.
                                Includes [Page N] markers between pages.
            page_count (int)  — Total number of pages in the PDF.
            is_scanned (bool) — True if >50% of pages required OCR.

    Raises:
        ValueError: If the bytes cannot be opened as a PDF.

    Example:
        text, pages, scanned = extract_text_from_pdf(file_bytes)
        print(f"Extracted {len(text)} chars from {pages} pages")
    """
    if not pdf_bytes:
        raise ValueError("pdf_bytes is empty — cannot extract text")

    try:
        doc        = fitz.open(stream=pdf_bytes, filetype="pdf")
        page_count = len(doc)

        if page_count == 0:
            doc.close()
            raise ValueError("PDF has no pages")

        logger.info(f"[pdf] Opened PDF: {page_count} page(s)")

        page_texts    = []
        scanned_pages = 0

        for page_num in range(page_count):
            page = doc.load_page(page_num)
            page_label = f"[Page {page_num + 1}]"

            # ── Strategy 1: native text layer ─────────────────────────────
            text = page.get_text("text")

            if not _is_page_empty(text):
                page_texts.append(f"{page_label}\n{text.strip()}")
                logger.debug(
                    f"[pdf] Page {page_num+1}: native text "
                    f"({len(text.strip())} chars)"
                )
                continue

            # ── Strategy 2: block-based extraction ────────────────────────
            blocks     = page.get_text("blocks")
            block_text = "\n".join(
                b[4].strip() for b in blocks
                if len(b) > 4 and isinstance(b[4], str) and b[4].strip()
            )

            if not _is_page_empty(block_text):
                page_texts.append(f"{page_label}\n{block_text}")
                logger.debug(
                    f"[pdf] Page {page_num+1}: block text "
                    f"({len(block_text)} chars)"
                )
                continue

            # ── Strategy 3: OCR for scanned pages ─────────────────────────
            logger.info(f"[pdf] Page {page_num+1}: no text found — attempting OCR")
            scanned_pages += 1
            ocr_text = _ocr_page(page)
            page_texts.append(f"{page_label}\n{ocr_text}")

        doc.close()

        # Assemble full text
        combined   = "\n\n".join(filter(None, page_texts))
        is_scanned = scanned_pages > (page_count / 2)

        logger.info(
            f"[pdf] Extraction complete: {len(combined):,} chars | "
            f"{page_count} pages | scanned_pages={scanned_pages} | "
            f"is_scanned={is_scanned}"
        )

        # Truncate to LLM context limit
        if len(combined) > _MAX_TEXT_LENGTH:
            logger.warning(
                f"[pdf] Text truncated from {len(combined):,} to "
                f"{_MAX_TEXT_LENGTH:,} chars (LLM context limit)"
            )
            combined = combined[:_MAX_TEXT_LENGTH] + "\n\n[... document truncated ...]"

        return combined, page_count, is_scanned

    except fitz.fitz.FitzError as exc:
        raise ValueError(f"PyMuPDF cannot open this PDF: {str(exc)}")
    except ValueError:
        raise
    except Exception as exc:
        logger.error(f"[pdf] Unexpected extraction error: {exc}", exc_info=True)
        raise ValueError(f"PDF extraction failed: {str(exc)}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN EXTRACTION — from file path
# ──────────────────────────────────────────────────────────────────────────────

def extract_text_from_file(file_path: str) -> Tuple[str, int, bool]:
    """
    Extract text from a PDF given its file system path.
    Convenience wrapper around extract_text_from_pdf().

    Args:
        file_path: Absolute or relative path to the PDF file.

    Returns:
        Same as extract_text_from_pdf(): (text, page_count, is_scanned)

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError:        If the path is not a PDF or cannot be opened.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: '{path.suffix}'")

    logger.info(f"[pdf] Reading file: {path} ({path.stat().st_size:,} bytes)")
    return extract_text_from_pdf(path.read_bytes())


# ──────────────────────────────────────────────────────────────────────────────
# OCR FALLBACK
# ──────────────────────────────────────────────────────────────────────────────

def _ocr_page(page: fitz.Page) -> str:
    """
    Rasterise a PDF page and apply tesseract OCR to extract text.
    Gracefully falls back if pytesseract or tesseract binary is not installed.

    Args:
        page: PyMuPDF Page object for the scanned page.

    Returns:
        Extracted OCR text string, or a placeholder message if OCR unavailable.
    """
    try:
        import pytesseract
        from PIL import Image

        # Rasterise page at _OCR_DPI for good character recognition quality
        scale_factor = _OCR_DPI / 72.0
        mat          = fitz.Matrix(scale_factor, scale_factor)
        pix          = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes    = pix.tobytes("png")

        # Convert to PIL Image
        image = Image.open(io.BytesIO(img_bytes))

        # OCR configuration: PSM 6 = assume single uniform block of text
        custom_config = f"--psm {_OCR_PSM} --oem 3"
        text = pytesseract.image_to_string(image, config=custom_config)

        if text.strip():
            logger.info(f"[pdf/ocr] OCR extracted {len(text.strip())} chars")
            return text.strip()
        else:
            logger.warning("[pdf/ocr] OCR returned empty text for this page")
            return "[OCR: page appears blank or contains only images]"

    except ImportError as exc:
        missing = "pytesseract" if "pytesseract" in str(exc) else "PIL"
        logger.warning(
            f"[pdf/ocr] {missing} not installed — OCR unavailable. "
            f"Install: pip install pytesseract Pillow"
            f"{'  +  sudo apt-get install tesseract-ocr' if 'pytesseract' in str(exc) else ''}"
        )
        return (
            "[Scanned page — OCR unavailable. "
            "Install pytesseract and tesseract-ocr for scanned document support.]"
        )

    except Exception as exc:
        logger.warning(f"[pdf/ocr] OCR failed for page: {exc}")
        return f"[OCR failed: {str(exc)[:100]}]"


# ──────────────────────────────────────────────────────────────────────────────
# TEXT CLEANING
# ──────────────────────────────────────────────────────────────────────────────

def clean_extracted_text(text: str) -> str:
    """
    Post-process raw extracted text for LLM consumption.

    Operations applied:
        1. Remove non-printable characters (keep tabs, newlines, ASCII printable)
        2. Collapse 3+ consecutive newlines → 2 newlines
        3. Collapse multiple spaces/tabs → single space
        4. Remove [Page N] markers added during extraction
        5. Strip leading/trailing whitespace

    Args:
        text: Raw text from extract_text_from_pdf() or _ocr_page().

    Returns:
        Cleaned text string ready for Gemini entity extraction.
    """
    if not text:
        return ""

    # 1. Remove non-printable characters (keep: tab=0x09, LF=0x0A, CR=0x0D, space-tilde)
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", " ", text)

    # 2. Normalise line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 3. Collapse 3+ consecutive newlines → exactly 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 4. Collapse multiple spaces/tabs on a single line → single space
    text = re.sub(r"[ \t]{2,}", " ", text)

    # 5. Remove [Page N] markers
    text = re.sub(r"\[Page \d+\]\n?", "", text)

    # 6. Remove OCR artefacts: lone single characters on a line
    text = re.sub(r"(?m)^[^\w\n]{1,2}$", "", text)

    # 7. Final trim
    return text.strip()


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _is_page_empty(text: str, min_chars: int = _MIN_CHARS_PER_PAGE) -> bool:
    """
    Determine whether extracted page text is effectively empty.

    A page is considered empty if the non-whitespace character count
    falls below min_chars.  This threshold filters out pages that contain
    only whitespace, form-feed characters, or very sparse headers.

    Args:
        text:      Raw extracted text string.
        min_chars: Minimum non-whitespace chars to be considered non-empty.

    Returns:
        True if the page is effectively empty, False otherwise.
    """
    if not text:
        return True
    cleaned = re.sub(r"\s+", "", text)
    return len(cleaned) < min_chars


def get_pdf_metadata(pdf_bytes: bytes) -> Dict[str, str]:
    """
    Extract PDF document metadata (title, author, dates, etc.).
    Used for document provenance tracking in MongoDB.

    Args:
        pdf_bytes: Raw PDF bytes.

    Returns:
        Dict with string values for all available metadata fields.
        Returns an empty dict on failure.
    """
    try:
        doc  = fitz.open(stream=pdf_bytes, filetype="pdf")
        meta = doc.metadata or {}
        page_count = len(doc)
        doc.close()

        return {
            "title":       meta.get("title",        ""),
            "author":      meta.get("author",        ""),
            "subject":     meta.get("subject",       ""),
            "creator":     meta.get("creator",       ""),
            "producer":    meta.get("producer",      ""),
            "created":     meta.get("creationDate",  ""),
            "modified":    meta.get("modDate",       ""),
            "page_count":  str(page_count),
            "encryption":  meta.get("encryption",    ""),
        }

    except Exception as exc:
        logger.warning(f"[pdf/meta] Metadata extraction failed: {exc}")
        return {}


def get_page_count(pdf_bytes: bytes) -> int:
    """
    Return the number of pages in a PDF without extracting text.
    Fast check before committing to full extraction.

    Args:
        pdf_bytes: Raw PDF bytes.

    Returns:
        Page count integer, or 0 on failure.
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        n   = len(doc)
        doc.close()
        return n
    except Exception:
        return 0


def validate_pdf(pdf_bytes: bytes) -> Tuple[bool, str]:
    """
    Validate that bytes represent a readable PDF file.

    Args:
        pdf_bytes: Raw bytes to validate.

    Returns:
        Tuple of (is_valid: bool, message: str).
        Message describes the problem if is_valid is False.
    """
    if not pdf_bytes:
        return False, "Empty file — no bytes received"

    # Check PDF magic bytes
    if not pdf_bytes[:5] == b"%PDF-":
        return False, "File does not start with PDF magic bytes (%PDF-)"

    try:
        doc        = fitz.open(stream=pdf_bytes, filetype="pdf")
        page_count = len(doc)
        doc.close()

        if page_count == 0:
            return False, "PDF has no pages"

        return True, f"Valid PDF with {page_count} page(s)"

    except fitz.fitz.FitzError as exc:
        return False, f"Corrupted or encrypted PDF: {str(exc)}"
    except Exception as exc:
        return False, f"PDF validation failed: {str(exc)}"