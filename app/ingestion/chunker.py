
from typing import TypedDict

from app.ingestion.pdf_loader import PageContent


class TextChunk(TypedDict):
    """A text chunk with source metadata."""

    source: str
    page_number: int
    chunk_index: int
    text: str


def chunk_pages(
    pages: list[PageContent],
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[TextChunk]:
    """Split page text into overlapping chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")

    chunks: list[TextChunk] = []

    for page in pages:
        text = page["text"].strip()

        if not text:
            continue

        start = 0
        chunk_index = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "source": page["source"],
                        "page_number": page["page_number"],
                        "chunk_index": chunk_index,
                        "text": chunk_text,
                    }
                )
                chunk_index += 1

            if end == len(text):
                break

            start = end - overlap

    return chunks
