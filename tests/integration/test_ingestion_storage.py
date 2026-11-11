
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import pymupdf
from streamlit import connection

from app.retrieval.retriever import Retriever
from app.storage.sqlite_store import get_document_chunks


class FakeEmbeddingModel:
    """Provide deterministic embeddings for integration testing."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [
            [1.0] + [0.0] * 383
            for _ in texts
        ]

    def embed_query(self, query: str) -> list[float]:
        return [1.0] + [0.0] * 383


class TestIngestionStorageIntegration(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        self.pdf_path = self.temp_path / "research.pdf"
        self.db_path = self.temp_path / "research_test.db"

        self.create_sample_pdf()

    def tearDown(self):
        self.temp_dir.cleanup()

    def create_sample_pdf(self):
        """Create a temporary PDF containing extractable text."""

        document = pymupdf.open()

        page = document.new_page()

        content = (
            "Artificial intelligence is transforming research. "
            "Machine learning helps systems identify patterns. "
            "Evidence-based research requires reliable information. "
        ) * 12

        page.insert_textbox(
            (50, 50, 550, 750),
            content,
            fontsize=11,
        )

        document.save(self.pdf_path)
        document.close()

    @patch("app.retrieval.retriever.EmbeddingModel", FakeEmbeddingModel)
    def test_pdf_ingestion_saves_to_sqlite_and_faiss(self):

        retriever = Retriever(db_path=self.db_path)

        chunk_count = retriever.ingest_pdf(self.pdf_path)

        # Verify chunks were generated
        self.assertGreater(chunk_count, 0)

        # Verify FAISS received the chunks
        self.assertEqual(
            len(retriever.store.chunks),
            chunk_count,
        )

        # Verify document was saved in SQLite
        with closing(sqlite3.connect(self.db_path)) as connection:
            document = connection.execute("SELECT id, filename FROM documents WHERE filename = ?",("research.pdf",),).fetchone()

        self.assertIsNotNone(document)

        document_id = document[0]

        # Verify chunks were saved in SQLite
        stored_chunks = get_document_chunks(
            document_id,
            db_path=self.db_path,
        )

        self.assertEqual(
            len(stored_chunks),
            chunk_count,
        )

        # Verify stored chunk content
        for chunk in stored_chunks:
            self.assertTrue(chunk["text"].strip())
            self.assertEqual(
                chunk["metadata"]["source"],
                "research.pdf",
            )

        # Verify FAISS retrieval works
        results = retriever.retrieve(
            "What is artificial intelligence?",
            top_k=3,
        )

        self.assertGreater(len(results), 0)

        print("\nIntegration test completed successfully.")
        print(f"Document ID: {document_id}")
        print(f"Chunks created: {chunk_count}")
        print(f"Chunks stored in SQLite: {len(stored_chunks)}")
        print(f"Chunks indexed in FAISS: {len(retriever.store.chunks)}")


if __name__ == "__main__":
    unittest.main()
