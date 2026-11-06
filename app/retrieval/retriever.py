
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
        self.store = VectorStore()

    def ingest_pdf(self, pdf_path: str | Path) -> int:
        """Extract, chunk, embed, store, and index a PDF."""

        pdf_path = Path(pdf_path)

        # 1. Extract text from PDF
        pages = extract_pdf_text(pdf_path)

        # 2. Split extracted text into chunks
        chunks = chunk_pages(pages)

        if not chunks:
            raise ValueError("No text chunks were created from the PDF.")

        # 3. Generate embeddings
        embeddings = self.embedder.embed_texts(
            [chunk["text"] for chunk in chunks]
        )

        # 4. Save document information in SQLite
        document_id = save_document(
            filename=pdf_path.name,
            file_path=str(pdf_path),
        )

        # 5. Save document chunks in SQLite
        save_chunks(document_id, chunks)

        # 6. Add chunks and embeddings to FAISS
        self.store.add(chunks, embeddings)

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
