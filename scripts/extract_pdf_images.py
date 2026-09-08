"""
Ultra-fast PDF embedded image extractor using PyMuPDF (fitz).
Extracts all figures, portraits, and heraldry plates from PDF files in seconds.
"""

import os
import fitz  # PyMuPDF
from pathlib import Path
from tqdm import tqdm


def extract_images_from_pdfs(source_dir: str, output_dir: str = "data/extracted_images"):
    source_path = Path(source_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not source_path.exists():
        print(f"[Error] Source directory not found: {source_path}")
        return

    pdf_files = list(source_path.rglob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files to process.")

    total_images_extracted = 0

    for pdf_file in tqdm(pdf_files, desc="Extracting Images", unit="pdf"):
        try:
            doc = fitz.open(str(pdf_file))
            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images(full=True)

                for img_idx, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]

                    # Filter out tiny icon artifacts (e.g. < 5KB)
                    if len(image_bytes) < 5000:
                        continue

                    img_filename = f"{pdf_file.stem}_p{page_num + 1}_img{img_idx + 1}.{image_ext}"
                    img_save_path = out_path / img_filename

                    with open(img_save_path, "wb") as f:
                        f.write(image_bytes)

                    total_images_extracted += 1

            doc.close()
        except Exception as e:
            print(f"Error extracting from {pdf_file.name}: {e}")

    print(f"\nExtraction complete! Saved {total_images_extracted} images to: {out_path.resolve()}")


if __name__ == "__main__":
    # Change this path to your PDF archive directory
    PDF_ARCHIVE_FOLDER = r"D:\projects\codefest-ai-challenge\Ashen_Era_Archive"
    extract_images_from_pdfs(PDF_ARCHIVE_FOLDER)
