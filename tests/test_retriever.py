
from app.retrieval.retriever import Retriever

retriever = Retriever()

results = retriever.retrieve(
    "What is machine learning?",
    top_k=3,
)

for i, (chunk, score) in enumerate(results, start=1):

    print(f"\n{'=' * 50}")
    print(f"EVIDENCE {i}")
    print(f"Similarity Score: {score:.4f}")

    print("\nMetadata:")

    for key, value in chunk.items():
        if key != "text":
            print(f"{key}: {value}")

    print("\nText:")
    print(chunk.get("text", "MISSING TEXT"))
    print()
