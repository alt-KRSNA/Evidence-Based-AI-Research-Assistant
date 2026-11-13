
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import pymupdf

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

        # Verify chunks were generated.
        self.assertGreater(chunk_count, 0)

        # Verify FAISS received the chunks.
        self.assertEqual(
            len(retriever.store.chunks),
            chunk_count,
        )

        # Verify document was saved in SQLite.
        with closing(sqlite3.connect(self.db_path)) as connection:
            document = connection.execute(
                "SELECT id, filename FROM documents WHERE filename = ?",
                ("research.pdf",),
            ).fetchone()

        self.assertIsNotNone(document)

        document_id = document[0]

        # Verify chunks were saved in SQLite.
        stored_chunks = get_document_chunks(
            document_id,
            db_path=self.db_path,
        )

        self.assertEqual(
            len(stored_chunks),
            chunk_count,
        )

        # Verify stored chunk content.
        for chunk in stored_chunks:
            self.assertTrue(chunk["text"].strip())
            self.assertEqual(
                chunk["metadata"]["source"],
                "research.pdf",
            )

        # Verify FAISS retrieval works.
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

    @patch("app.retrieval.retriever.EmbeddingModel", FakeEmbeddingModel)
    def test_duplicate_pdf_ingestion_does_not_create_duplicate_chunks(self):

        retriever = Retriever(db_path=self.db_path)

        # First upload.
        first_count = retriever.ingest_pdf(self.pdf_path)

        initial_faiss_count = len(retriever.store.chunks)

        # Second upload of the exact same PDF.
        second_count = retriever.ingest_pdf(self.pdf_path)

        # Verify second ingestion adds no chunks.
        self.assertEqual(second_count, 0)

        # Verify FAISS vector count remains unchanged.
        self.assertEqual(
            len(retriever.store.chunks),
            initial_faiss_count,
        )

        # Verify SQLite contains only one document.
        with closing(sqlite3.connect(self.db_path)) as connection:
            document_count = connection.execute(
                "SELECT COUNT(*) FROM documents"
            ).fetchone()[0]

            chunk_count = connection.execute(
                "SELECT COUNT(*) FROM chunks"
            ).fetchone()[0]

        self.assertEqual(document_count, 1)
        self.assertEqual(chunk_count, first_count)

        print("\nDuplicate ingestion test passed.")


    
    @patch("app.retrieval.retriever.EmbeddingModel", FakeEmbeddingModel)
    def test_faiss_index_recovers_from_sqlite_when_files_are_missing(self):

        # First ingestion creates SQLite records and FAISS files.
        retriever = Retriever(db_path=self.db_path)

        original_count = retriever.ingest_pdf(self.pdf_path)

        self.assertGreater(original_count, 0)

        index_path = retriever.index_path

        index_file = index_path / "index.faiss"
        metadata_file = index_path / "chunks.json"

        self.assertTrue(index_file.exists())
        self.assertTrue(metadata_file.exists())

        # Simulate losing the FAISS index.
        index_file.unlink()
        metadata_file.unlink()

        self.assertFalse(index_file.exists())
        self.assertFalse(metadata_file.exists())

        # Create a new Retriever. It should rebuild from SQLite.
        recovered_retriever = Retriever(
            db_path=self.db_path
        )

        # Verify every chunk was recovered.
        self.assertEqual(
            len(recovered_retriever.store.chunks),
            original_count,
        )

        # Verify both FAISS files were recreated.
        self.assertTrue(index_file.exists())
        self.assertTrue(metadata_file.exists())

        # Verify retrieval works after recovery.
        results = recovered_retriever.retrieve(
            "What is artificial intelligence?",
            top_k=3,
        )

        self.assertGreater(len(results), 0)

        print("\nFAISS recovery test passed.")
        print(f"Recovered vectors: {len(recovered_retriever.store.chunks)}")


    @patch("app.retrieval.retriever.EmbeddingModel", FakeEmbeddingModel)
    def test_identical_chunks_from_different_documents_remain_isolated(self):

        retriever = Retriever(db_path=self.db_path)

        # Ingest the first PDF.
        first_count = retriever.ingest_pdf(self.pdf_path)

        first_document_id = retriever.store.chunks[0]["document_id"]

        # Create another PDF with identical extracted text,
        # but different metadata and a different file hash.
        second_dir = self.temp_path / "second"
        second_dir.mkdir()

        second_pdf_path = second_dir / "research.pdf"

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

        document.set_metadata(
            {"title": "Second Research Document"}
        )

        document.save(second_pdf_path)
        document.close()

        # Ingest the second PDF.
        second_count = retriever.ingest_pdf(second_pdf_path)

        # Verify both PDFs generated chunks.
        self.assertEqual(second_count, first_count)

        # Verify FAISS contains chunks from both documents.
        self.assertEqual(
            len(retriever.store.chunks),
            first_count + second_count,
        )

        second_document_id = next(
            chunk["document_id"]
            for chunk in retriever.store.chunks
            if chunk["document_id"] != first_document_id
        )

        # Verify document IDs are different.
        self.assertNotEqual(
            first_document_id,
            second_document_id,
        )

        # Verify both documents are represented in FAISS.
        indexed_document_ids = {
            chunk["document_id"]
            for chunk in retriever.store.chunks
        }

        self.assertEqual(
            indexed_document_ids,
            {first_document_id, second_document_id},
        )

        # Verify SQLite contains two separate documents.
        with closing(sqlite3.connect(self.db_path)) as connection:
            document_count = connection.execute(
                "SELECT COUNT(*) FROM documents"
            ).fetchone()[0]

            chunk_count = connection.execute(
                "SELECT COUNT(*) FROM chunks"
            ).fetchone()[0]

        self.assertEqual(document_count, 2)
        self.assertEqual(
            chunk_count,
            first_count + second_count,
        )

        print("\nDocument isolation test passed.")
        print(f"First document ID: {first_document_id}")
        print(f"Second document ID: {second_document_id}")
        print(f"Total FAISS chunks: {len(retriever.store.chunks)}")


    @patch("app.retrieval.retriever.EmbeddingModel", FakeEmbeddingModel)
    def test_faiss_index_recovers_when_index_file_is_corrupted(self):

        # Create a valid index first.
        retriever = Retriever(db_path=self.db_path)

        original_count = retriever.ingest_pdf(self.pdf_path)

        index_file = retriever.index_path / "index.faiss"
        metadata_file = retriever.index_path / "chunks.json"

        self.assertTrue(index_file.exists())
        self.assertTrue(metadata_file.exists())

        # Simulate a corrupted FAISS index.
        index_file.write_bytes(b"This is a corrupted FAISS index.")

        # Restart Retriever to trigger automatic recovery.
        recovered_retriever = Retriever(
            db_path=self.db_path
        )

        # Check that all chunks were recovered.
        self.assertEqual(
            len(recovered_retriever.store.chunks),
            original_count,
        )

        # Check that the index files were recreated.
        self.assertTrue(index_file.exists())
        self.assertTrue(metadata_file.exists())

        # Check that retrieval works after recovery.
        results = recovered_retriever.retrieve(
            "What is artificial intelligence?",
            top_k=3,
        )

        self.assertGreater(len(results), 0)

        print("\nCorrupted FAISS recovery test passed.")
        print(f"Recovered vectors: {len(recovered_retriever.store.chunks)}")


if __name__ == "__main__":
    unittest.main()
