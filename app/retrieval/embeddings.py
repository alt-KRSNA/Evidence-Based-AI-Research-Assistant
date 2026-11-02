
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingModel:
    """Convert text into numerical vectors for semantic search."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Convert multiple text strings into embedding vectors."""

        if not texts:
            return []

        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Convert one user question into an embedding vector."""

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        return self.embed_texts([query])[0]
