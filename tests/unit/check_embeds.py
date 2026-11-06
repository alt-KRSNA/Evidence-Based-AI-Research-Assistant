from app.retrieval.embeddings import EmbeddingModel

embedder = EmbeddingModel()

texts = [
    "Python developer with machine learning experience",
    "Engineer working on artificial intelligence",
]

vectors = embedder.embed_texts(texts)

print("Number of vectors:", len(vectors))
print("Dimensions per vector:", len(vectors[0]))
print("First 5 values:", vectors[0][:5])