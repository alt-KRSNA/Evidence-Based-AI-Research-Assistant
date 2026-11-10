
from app.pipeline import ResearchPipeline


pipeline = ResearchPipeline()

question = "What is machine learning?"

result = pipeline.ask(
    question=question,
    top_k=3,
)

print("\nQUESTION:")
print(result["question"])

print("\nGENERATED ANSWER:")
print(result["answer"])

print("\nRETRIEVED EVIDENCE:")

for index, (chunk, score) in enumerate(
    result["evidence"], start=1
):
    print(f"\nEvidence {index} | Score: {score:.4f}")
    print("Source:", chunk.get("source", "Unknown"))
    print("Page:", chunk.get("page", "Unknown"))
    print("Text:", chunk["text"][:300])
