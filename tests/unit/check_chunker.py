
from app.ingestion.pdf_loader import extract_pdf_text
from app.ingestion.chunker import chunk_pages

pdf_path = r"sample.pdf"

pages = extract_pdf_text(pdf_path)
chunks = chunk_pages(pages)

print(f"Pages extracted: {len(pages)}")
print(f"Chunks created: {len(chunks)}")

for chunk in chunks[:3]:
    print(f"\nChunk {chunk['chunk_index']}")
    print(f"Source: {chunk['source']}")
    print(f"Page: {chunk['page_number']}")
    print(f"Text: {chunk['text'][:200]}")
