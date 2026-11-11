
from collections import Counter
from pathlib import Path

from app.ingestion.chunker import chunk_pages
from app.ingestion.pdf_loader import extract_pdf_text
from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.vector_store import VectorStore
from app.storage.sqlite_store import (
    DEFAULT_DB_PATH,
    initialize_database,
    calculate_file_hash,
    get_document_by_hash,
    get_document_chunks,
    save_document,
    save_chunks,
)


class Retriever:
    """Retrieve relevant evidence from ingested PDFs."""

    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB_PATH,
        index_path: str | Path | None = None,
    ) -> None:

        self.db_path = Path(db_path)

        # Keep test databases and indexes isolated.
        if index_path is not None:
            self.index_path = Path(index_path)
        elif self.db_path == DEFAULT_DB_PATH:
            self.index_path = Path("data/indexes")
        else:
            self.index_path = self.db_path.parent / "indexes"

        initialize_database(self.db_path)

        self.embedder = EmbeddingModel()

        if (
            (self.index_path / "index.faiss").exists()
            and (self.index_path / "chunks.json").exists()
        ):
            self.store = VectorStore.load(self.index_path)
        else:
            self.store = VectorStore()

    @staticmethod
    def _chunk_key(chunk: dict) -> tuple:
        """Create a unique identity for a chunk."""

        return (
            chunk["source"],
            chunk["page_number"],
            chunk["chunk_index"],
            chunk["text"],
        )

    @staticmethod
    def _prepare_index_chunks(
        chunks: list[dict],
        document_id: str,
    ) -> list[dict]:
        """Attach document IDs to chunks stored in FAISS."""

        return [
            {
                **chunk,
                "document_id": document_id,
            }
            for chunk in chunks
        ]

    def _sync_existing_document(
        self,
        document_id: str,
    ) -> int:
        """Ensure an existing document's chunks are present in FAISS."""

        db_chunks = get_document_chunks(
            document_id,
            db_path=self.db_path,
        )

        if not db_chunks:
            raise ValueError(
                "Document exists in SQLite but has no stored chunks."
            )

        expected_chunks = [
            {
                "source": chunk["metadata"]["source"],
                "page_number": chunk["page_number"],
                "chunk_index": chunk["metadata"]["chunk_index"],
                "text": chunk["text"],
            }
            for chunk in db_chunks
        ]

        expected_counts = Counter(
            self._chunk_key(chunk)
            for chunk in expected_chunks
        )

        indexed_counts = Counter(
            self._chunk_key(chunk)
            for chunk in self.store.chunks
        )

        missing_chunks = []

        for chunk in expected_chunks:
            key = self._chunk_key(chunk)

            if indexed_counts[key] < expected_counts[key]:
                missing_chunks.append(chunk)
                indexed_counts[key] += 1

        if not missing_chunks:
            print("Document already indexed. Skipping ingestion.")
            return 0

        print(
            f"Found {len(missing_chunks)} missing chunks. "
            "Synchronizing FAISS..."
        )

        embeddings = self.embedder.embed_texts(
            [chunk["text"] for chunk in missing_chunks]
        )

        index_chunks = self._prepare_index_chunks(
            missing_chunks,
            document_id,
        )

        self.store.add(index_chunks, embeddings)
        self.store.save(self.index_path)

        print("FAISS synchronization completed.")

        return len(missing_chunks)

    def ingest_pdf(self, pdf_path: str | Path) -> int:
        """Ingest a PDF without creating duplicate vectors."""

        pdf_path = Path(pdf_path)

        if not pdf_path.is_file():
            raise FileNotFoundError(
                f"PDF file not found: {pdf_path}"
            )

        # Step 1: Identify the PDF using its content hash.
        file_hash = calculate_file_hash(pdf_path)

        # Step 2: Check whether this PDF already exists.
        existing_document = get_document_by_hash(
            file_hash,
            db_path=self.db_path,
        )

        if existing_document:
            print(
                f"Document already exists: "
                f"{existing_document['filename']}"
            )

            return self._sync_existing_document(
                existing_document["id"]
            )

        # Step 3: Extract and chunk a new PDF.
        pages = extract_pdf_text(pdf_path)
        chunks = chunk_pages(pages)

        if not chunks:
            raise ValueError(
                "No text chunks were created from the PDF."
            )

        # Step 4: Generate embeddings.
        embeddings = self.embedder.embed_texts(
            [chunk["text"] for chunk in chunks]
        )

        # Step 5: Register the document.
        document_id = save_document(
            filename=pdf_path.name,
            file_path=str(pdf_path.resolve()),
            file_hash=file_hash,
            db_path=self.db_path,
        )

        # Step 6: Save chunks in SQLite.
        save_chunks(
            document_id,
            chunks,
            db_path=self.db_path,
        )

        # Step 7: Attach document IDs to FAISS metadata.
        index_chunks = self._prepare_index_chunks(
            chunks,
            document_id,
        )

        # Step 8: Add vectors and persist the index.
        self.store.add(index_chunks, embeddings)
        self.store.save(self.index_path)

        print(f"Document ingested successfully: {pdf_path.name}")
        print(f"Chunks indexed: {len(chunks)}")

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
