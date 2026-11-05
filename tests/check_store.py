
import tempfile
import unittest

from pathlib import Path

from app.storage.sqlite_store import (
    initialize_database,
    save_document,
    save_chunks,
    get_chunk,
    get_document_chunks,
)

class TestSQLiteStorage(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test.db"

        initialize_database(self.db_path)

        self.document_id = save_document(
            filename="AI_Sample.pdf",
            file_path="data/raw/AI_Sample.pdf",
            db_path=self.db_path
        )

        self.sample_chunks = [
            {
                "chunk_id": "chunk_0",
                "page_number": 1,
                "text": "Introduction to Artificial Intelligence.",
                "token_count": 8
            },
            {
                "chunk_id": "chunk_1",
                "page_number": 1,
                "text": "Machine learning is a subset of AI.",
                "token_count": 9
            }
        ]

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_chunks(self):
        count = save_chunks(
            self.document_id,
            self.sample_chunks,
            self.db_path
        )

        self.assertEqual(count, 2)

    def test_retrieve_chunk(self):
        save_chunks(
            self.document_id,
            self.sample_chunks,
            self.db_path
        )

        chunk = get_chunk(
            self.document_id,
            "chunk_0",
            self.db_path
        )

        self.assertIsNotNone(chunk)
        self.assertEqual(
            chunk["text"],
            "Introduction to Artificial Intelligence."
        )

    def test_retrieve_all_chunks(self):
        save_chunks(
            self.document_id,
            self.sample_chunks,
            self.db_path
        )

        chunks = get_document_chunks(
            self.document_id,
            self.db_path
        )

        self.assertEqual(len(chunks), 2)

    def test_document_isolation(self):
        save_chunks(
            self.document_id,
            self.sample_chunks,
            self.db_path
        )

        chunks = get_document_chunks(
            "non_existing_document",
            self.db_path
        )

        self.assertEqual(chunks, [])

    def test_update_existing_chunk(self):
        save_chunks(
            self.document_id,
            self.sample_chunks,
            self.db_path
        )

        updated_chunks = [{
            **self.sample_chunks[0],
            "text": "Updated AI introduction."
        }]

        save_chunks(
            self.document_id,
            updated_chunks,
            self.db_path
        )

        chunk = get_chunk(
            self.document_id,
            "chunk_0",
            self.db_path
        )

        self.assertEqual(
            chunk["text"],
            "Updated AI introduction."
        )

if __name__ == "__main__":
    unittest.main()


