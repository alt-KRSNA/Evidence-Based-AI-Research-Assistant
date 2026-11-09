
from pathlib import Path

from app.ingestion.chunker import chunk_pages
from app.ingestion.pdf_loader import extract_pdf_text
from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.vector_store import VectorStore
from app.storage.sqlite_store import (
    initialize_database,
    save_document,
    save_chunks,
)


class Retriever:
    """Retrieve relevant evidence from an ingested PDF."""

    def __init__(self) -> None:
        initialize_database()

        self.embedder = EmbeddingModel()

        self.index_path = Path("data/indexes")

        if (
            (self.index_path / "index.faiss").exists()
            and (self.index_path / "chunks.json").exists()
        ):
            self.store = VectorStore.load(self.index_path)
        else:
            self.store = VectorStore()

    def ingest_pdf(self, pdf_path: str | Path) -> int:
        """Extract, chunk, embed, store, and index a PDF."""

        pdf_path = Path(pdf_path)

        pages = extract_pdf_text(pdf_path)
        chunks = chunk_pages(pages)

        if not chunks:
            raise ValueError("No text chunks were created from the PDF.")

        embeddings = self.embedder.embed_texts(
            [chunk["text"] for chunk in chunks]
        )

        document_id = save_document(
            filename=pdf_path.name,
            file_path=str(pdf_path),
        )

        save_chunks(document_id, chunks)

        self.store.add(chunks, embeddings)

        self.store.save(self.index_path)

        return len(chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[tuple[dict, float]]:
        """Return relevant chunks and their similarity scores."""

        query_embedding = self.embedder.embed_query(query)

        return self.store.search(
            query_embedding,
            top_k=top_k,
        )
