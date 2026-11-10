
import json
from collections import Counter
from pathlib import Path

index_file = Path("data/indexes/chunks.json")

with index_file.open("r", encoding="utf-8") as file:
    chunks = json.load(file)

keys = [
    (
        chunk["source"],
        chunk["page_number"],
        chunk["chunk_index"],
    )
    for chunk in chunks
]

counts = Counter(keys)

duplicates = {
    key: count
    for key, count in counts.items()
    if count > 1
}

print("Total stored chunks:", len(chunks))
print("Unique chunks:", len(counts))
print("Duplicate records:", sum(duplicates.values()))

for key, count in duplicates.items():
    print(f"{key} -> appears {count} times")
