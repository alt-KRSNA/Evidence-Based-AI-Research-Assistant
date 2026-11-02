
from app.retrieval.retriever import Retriever

retriever = Retriever()

pdf_path = r"sample.pdf"

count = retriever.ingest_pdf(pdf_path)
print(f"Indexed {count} chunks.")

query = input("\nAsk a question about your PDF: ")
results = retriever.retrieve(query, top_k=3)

for chunk, score in results:
    print(f"\nSimilarity: {score:.4f}")
    print(f"Source: {chunk['source']}, page {chunk['page_number']}")
    print(chunk["text"])
