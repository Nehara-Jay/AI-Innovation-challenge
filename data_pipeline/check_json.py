import json
from collections import Counter

# Load your extracted data
with open("extracted_archive.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 1. Count total unique files processed
unique_files = set(chunk["source"] for chunk in data)
print(f"Total unique files processed: {len(unique_files)} (Target: 415)")

# 2. Count total chunks (should be 329 based on your terminal)
print(f"Total chunks extracted: {len(data)}")

# 3. See exactly what types of files were processed
file_types = Counter(chunk["source"].split(".")[-1].lower() for chunk in data)
print("\nBreakdown by file extension:")
for ext, count in file_types.items():
    print(f" .{ext}: {count} chunks")

# 4. (Optional) Print all unique filenames to see what folder you actually processed
print("\nFirst 10 files processed:")
for file in list(unique_files)[:10]:
    print(f" - {file}")