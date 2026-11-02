
from pathlib import Path

from app.ingestion.pdf_loader import extract_pdf_text


pdf_path = Path(input("Enter the path to a PDF file: ").strip().strip('"'))

try:
    pages = extract_pdf_text(pdf_path)

    print(f"\nExtracted {len(pages)} pages.")

    for page in pages[:2]:
        print(f"\nSource: {page['source']}")
        print(f"Page: {page['page_number']}")
        print(f"Text preview: {page['text'][:300]}")

except (FileNotFoundError, ValueError, RuntimeError) as exc:
    print(f"PDF extraction failed: {exc}")
