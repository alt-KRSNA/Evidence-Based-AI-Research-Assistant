
from pathlib import Path
from typing import TypedDict

import pymupdf


class PageContent(TypedDict):
    """Text and source metadata extracted from one PDF page."""

    source: str
    page_number: int
    text: str


def extract_pdf_text(pdf_path: str | Path) -> list[PageContent]:
    """Extract text from each page of a PDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        A list of page records, including source filename,
        one-based page number, and extracted text.

    Raises:
        FileNotFoundError: If the PDF does not exist.
        ValueError: If the path is not a PDF or the document
            contains no extractable text.
        RuntimeError: If the PDF cannot be opened or processed.
    """
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    if not path.is_file():
        raise ValueError(f"Expected a PDF file, got: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.name}")

    pages: list[PageContent] = []

    try:
        with pymupdf.open(path) as document:
            if document.is_encrypted:
                raise ValueError("Encrypted PDFs are not supported.")

            for page_index, page in enumerate(document):
                text = page.get_text("text").strip()

                pages.append(
                    {
                        "source": path.name,
                        "page_number": page_index + 1,
                        "text": text,
                    }
                )

    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Failed to process PDF '{path.name}': {exc}"
        ) from exc

    if not any(page["text"] for page in pages):
        raise ValueError(
            f"No extractable text found in PDF: {path.name}. "
            "The PDF may be scanned or empty."
        )

    return pages
