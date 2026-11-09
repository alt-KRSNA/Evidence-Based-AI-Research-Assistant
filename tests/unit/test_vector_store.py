import tempfile
import unittest
from pathlib import Path

from app.retrieval.vector_store import VectorStore


class TestVectorStorePersistence(unittest.TestCase):

    def test_save_and_load(self):
        store = VectorStore(dimension=3)

        chunks = [
            {
                "source": "research.pdf",
                "page_number": 1,
                "chunk_index": 0,
                "text": "Artificial intelligence is transforming research."
            }
        ]

        embeddings = [[1.0, 0.0, 0.0]]

        store.add(chunks, embeddings)

        with tempfile.TemporaryDirectory() as temp_dir:
            store.save(temp_dir)

            loaded_store = VectorStore.load(temp_dir)

            self.assertEqual(len(loaded_store.chunks), 1)
            self.assertEqual(
                loaded_store.chunks[0]["text"],
                chunks[0]["text"]
            )

            results = loaded_store.search(
                [1.0, 0.0, 0.0],
                top_k=1
            )

            self.assertEqual(len(results), 1)

    def test_saved_files_exist(self):
        store = VectorStore(dimension=3)

        with tempfile.TemporaryDirectory() as temp_dir:
            store.add(
                [{"source": "test.pdf", "page_number": 1,
                  "chunk_index": 0, "text": "Test data"}],
                [[1.0, 0.0, 0.0]]
            )

            store.save(temp_dir)

            self.assertTrue(Path(temp_dir, "index.faiss").exists())
            self.assertTrue(Path(temp_dir, "chunks.json").exists())


if __name__ == "__main__":
    unittest.main()