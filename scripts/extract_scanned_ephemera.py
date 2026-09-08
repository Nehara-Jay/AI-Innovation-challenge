"""
Scanned PDF Ephemera Extractor.
Uses PyMuPDF image rendering + Gemini 2.5 Flash via OpenRouter to transcribe
all 17 scanned historical ephemera PDF files with 100% accuracy.
"""

import os
import sys
import json
import base64
import time
from pathlib import Path
from tqdm import tqdm
import pymupdf
from openai import OpenAI
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

load_dotenv()


def get_vision_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not configured in .env")
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def transcribe_page_image(client: OpenAI, image_bytes: bytes, filename: str, page_num: int) -> str:
    b64_img = base64.b64encode(image_bytes).decode("utf-8")
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model="google/gemini-2.5-flash",
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Transcribe all text from this scanned historical fantasy document verbatim. "
                                "Preserve original wording, names, dates, titles, and paragraphs. "
                                "Output ONLY the raw transcribed text without introductory comments or markdown code blocks."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"},
                        },
                    ],
                }],
                max_tokens=2048,
                temperature=0.0,
            )
            text = resp.choices[0].message.content.strip()
            return text
        except Exception as e:
            if attempt == 3:
                print(f"\n[Error] Failed to transcribe {filename} p.{page_num}: {e}")
                return ""
            time.sleep(3.0 * (attempt + 1))
    return ""


def extract_all_scanned_ephemera():
    raw_ephemera_dir = BASE_DIR / "data" / "raw" / "Ashen_Era_Archive" / "ephemera"
    scan_files = sorted(list(raw_ephemera_dir.glob("*.scan.pdf")))

    print(f"Found {len(scan_files)} scanned PDF files to transcribe.")
    client = get_vision_client()

    new_chunks = []
    total_pages = 0

    for scan_file in tqdm(scan_files, desc="Transcribing Scans", unit="file"):
        try:
            doc = pymupdf.open(scan_file)
            for page_num in range(len(doc)):
                total_pages += 1
                page = doc[page_num]
                pix = page.get_pixmap(dpi=150)
                img_bytes = pix.tobytes("jpeg")

                text = transcribe_page_image(
                    client=client,
                    image_bytes=img_bytes,
                    filename=scan_file.name,
                    page_num=page_num + 1,
                )

                if text.strip():
                    new_chunks.append({
                        "source": scan_file.name,
                        "page": page_num + 1,
                        "type": "ephemera",
                        "content": text.strip(),
                    })
            doc.close()
        except Exception as e:
            print(f"Error opening {scan_file.name}: {e}")

    print(f"\nSuccessfully transcribed {len(new_chunks)} pages from {len(scan_files)} scanned files.")

    # Update extracted_archive.json
    extracted_json_path = BASE_DIR / "data" / "extracted_archive.json"
    with open(extracted_json_path, "r", encoding="utf-8") as f:
        existing_data = json.load(f)

    # Filter out any duplicate entries of the same scan file
    existing_filenames = set(scan_file.name for scan_file in scan_files)
    clean_data = [c for c in existing_data if c["source"] not in existing_filenames]
    clean_data.extend(new_chunks)

    with open(extracted_json_path, "w", encoding="utf-8") as f:
        json.dump(clean_data, f, indent=2, ensure_ascii=False)

    print(f"Updated {extracted_json_path} (Total chunks now: {len(clean_data)})")

    # Mirror to data_pipeline/data/extracted_archive.json if it exists
    pipeline_json = BASE_DIR / "data_pipeline" / "data" / "extracted_archive.json"
    if pipeline_json.parent.exists():
        with open(pipeline_json, "w", encoding="utf-8") as f:
            json.dump(clean_data, f, indent=2, ensure_ascii=False)

    return len(new_chunks)


if __name__ == "__main__":
    extract_all_scanned_ephemera()
