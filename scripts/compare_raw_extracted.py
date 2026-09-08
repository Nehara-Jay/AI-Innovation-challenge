"""
Script to deeply compare data/raw/Ashen_Era_Archive with data/extracted_archive.json.
"""

import sys
import json
from pathlib import Path
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
raw_dir = BASE_DIR / "data" / "raw" / "Ashen_Era_Archive"
extracted_file = BASE_DIR / "data" / "extracted_archive.json"

if not raw_dir.exists():
    print(f"Error: Raw directory not found at {raw_dir}")
    sys.exit(1)

with open(extracted_file, "r", encoding="utf-8") as f:
    extracted_data = json.load(f)

# 1. Scan Raw Files
raw_files = []
for p in raw_dir.rglob("*"):
    if p.is_file():
        rel = p.relative_to(raw_dir)
        folder = p.parent.name if p.parent != raw_dir else "root"
        raw_files.append({
            "rel_path": str(rel),
            "filename": p.name,
            "ext": p.suffix.lower(),
            "size": p.stat().st_size,
            "folder": folder,
        })

print("=" * 75)
print(f"📁 RAW ARCHIVE INVENTORY (Total Files: {len(raw_files)})")
print("=" * 75)

folder_counts = Counter(f["folder"] for f in raw_files)
for folder, count in folder_counts.most_common():
    print(f"  📂 {folder:<15}: {count:>3} files")

ext_counts = Counter(f["ext"] for f in raw_files)
print("\nExtensions in raw archive:")
for ext, count in ext_counts.most_common():
    print(f"  {ext:<8}: {count:>3} files")

# 2. Extracted Dataset
extracted_sources = set(c["source"] for c in extracted_data)
extracted_by_type = Counter(c.get("type", "document") for c in extracted_data)

print("\n" + "=" * 75)
print(f"📑 EXTRACTED DATASET INVENTORY (Total Chunks: {len(extracted_data)})")
print("=" * 75)
print(f"Unique source filenames in extracted_archive.json: {len(extracted_sources)}")
for t, count in extracted_by_type.items():
    print(f"  Tag '{t}': {count} chunks")

# 3. Exact Comparison
raw_filenames = {f["filename"]: f for f in raw_files}
extracted_filenames = set(extracted_sources)

matched = [f for f in raw_files if f["filename"] in extracted_filenames]
unextracted = [f for f in raw_files if f["filename"] not in extracted_filenames]

print("\n" + "=" * 75)
print(f"🔍 EXACT MATCH & GAP ANALYSIS")
print("=" * 75)
print(f"✅ Successfully Extracted Files : {len(matched):>3} / {len(raw_files)}")
print(f"⚠️ Unextracted / Skipped Files  : {len(unextracted):>3} / {len(raw_files)}")

if unextracted:
    print("\nBreakdown of Unextracted / Skipped Files by Folder & Type:")
    unext_counter = Counter(f"{f['folder']} ({f['ext']})" for f in unextracted)
    for cat, count in unext_counter.most_common():
        print(f"  - {cat:<22}: {count:>3} files")

    print("\nDetailed list of unextracted files:")
    for f in unextracted:
        print(f"  • [{f['folder']:<9}] {f['filename']:<45} ({f['size']:>8,} bytes)")
