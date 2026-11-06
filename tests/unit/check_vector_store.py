
from app.ingestion.chunker import TextChunk
from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.vector_store import VectorStore

chunks: list[TextChunk] = [
    {
        "source": "test.pdf",
        "page_number": 1,
        "chunk_index": 0,
        "text": "Python is a programming language used for machine learning.",
    },
    {
        "source": "test.pdf",
        "page_number": 2,
        "chunk_index": 0,
        "text": "The capital of France is Paris.",
    },
    {
        "source": "test.pdf",
        "page_number": 3,
        "chunk_index": 0,
        "text": "Neural networks are used in deep learning.",
    },
]

embedder = EmbeddingModel()
vectors = embedder.embed_texts([chunk["text"] for chunk in chunks])

store = VectorStore()
store.add(chunks, vectors)

query = "Which language is used for machine learning?"
query_vector = embedder.embed_query(query)

results = store.search(query_vector, top_k=2)

print(f"Query: {query}")
print("\nTop matches:")

for chunk, score in results:
    print(f"\nSimilarity score: {score:.4f}")
    print(f"Source: {chunk['source']}, page {chunk['page_number']}")
    print(f"Text: {chunk['text']}")
