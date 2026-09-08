import json
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent.parent
data_file = BASE_DIR / "data" / "extracted_archive.json"

if not data_file.exists():
    print(f"File not found: {data_file}")
    exit(1)

with open(data_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# 1. Count total unique files processed
unique_files = set(chunk["source"] for chunk in data)
print(f"Total unique files processed: {len(unique_files)} (Competition Target: ~415)")

# 2. Count total chunks
print(f"Total raw chunks extracted: {len(data)}")

# 3. See breakdown of file types
file_types = Counter(chunk["source"].split(".")[-1].lower() for chunk in data)
print("\nBreakdown by file extension:")
for ext, count in file_types.most_common():
    print(f"  .{ext}: {count} chunks")

# 4. Check if chunked data file exists
chunked_file = BASE_DIR / "data" / "chunked_archive.json"
if chunked_file.exists():
    with open(chunked_file, "r", encoding="utf-8") as f:
        chunked_data = json.load(f)
    print(f"\nNormalized RAG chunks: {len(chunked_data)}")
    lengths = [c.get("char_count", len(c.get("content", ""))) for c in chunked_data]
    print(f"  Max chars: {max(lengths)}, Avg chars: {sum(lengths)/len(lengths):.1f}")