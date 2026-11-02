
from pathlib import Path

from app.ingestion.chunker import chunk_pages
from app.ingestion.pdf_loader import extract_pdf_text
from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.vector_store import VectorStore


class Retriever:
    """Retrieve relevant evidence from an ingested PDF."""

    def __init__(self) -> None:
        self.embedder = EmbeddingModel()
        self.store = VectorStore()

    def ingest_pdf(self, pdf_path: str | Path) -> int:
        """Extract, chunk, embed, and index a PDF."""

        pages = extract_pdf_text(pdf_path)
        chunks = chunk_pages(pages)

        if not chunks:
            raise ValueError("No text chunks were created from the PDF.")

        embeddings = self.embedder.embed_texts(
            [chunk["text"] for chunk in chunks]
        )

        self.store.add(chunks, embeddings)

        return len(chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[tuple[dict, float]]:
        """Return relevant chunks and their similarity scores."""

        query_embedding = self.embedder.embed_query(query)
        return self.store.search(query_embedding, top_k=top_k)
