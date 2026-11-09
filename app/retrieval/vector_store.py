
import faiss
import numpy as np
import json
from pathlib import Path

from app.ingestion.chunker import TextChunk


class VectorStore:
    """Store chunk embeddings and retrieve similar chunks."""

    def __init__(self, dimension: int = 384) -> None:
        if dimension <= 0:
            raise ValueError("Dimension must be positive.")

        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: list[TextChunk] = []
        self.dimension = dimension

    def add(
        self,
        chunks: list[TextChunk],
        embeddings: list[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have one embedding.")

        if not chunks:
            return

        vectors = np.asarray(embeddings, dtype=np.float32)

        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embeddings with {self.dimension} dimensions."
            )

        if not np.isfinite(vectors).all():
            raise ValueError("Embeddings contain invalid numeric values.")

        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 3,
    ) -> list[tuple[TextChunk, float]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        if not self.chunks:
            return []

        query_vector = np.asarray([query_embedding], dtype=np.float32)

        if query_vector.shape != (1, self.dimension):
            raise ValueError(
                f"Query embedding must have {self.dimension} dimensions."
            )

        if not np.isfinite(query_vector).all():
            raise ValueError("Query embedding contains invalid values.")

        scores, indices = self.index.search(
            query_vector,
            min(top_k, len(self.chunks)),
        )

        results = []

        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue

            results.append(
                (self.chunks[int(index)], float(score))
            )

        return results

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(
            self.index,
            str(directory / "index.faiss")
        )

        with open(directory / "chunks.json", "w", encoding="utf-8") as file:
            json.dump(self.chunks, file, indent=2)

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        directory = Path(directory)

        index = faiss.read_index(str(directory / "index.faiss"))

        with open(directory / "chunks.json", "r", encoding="utf-8") as file:
            chunks = json.load(file)

        if index.ntotal != len(chunks):
            raise ValueError("FAISS index and chunk metadata do not match.")

        store = cls(dimension=index.d)
        store.index = index
        store.chunks = chunks

        return store
