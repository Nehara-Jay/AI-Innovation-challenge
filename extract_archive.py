import os
import json
import fitz  # PyMuPDF for lightning-fast PDF text extraction
import docx  # python-docx for Word files
from paddleocr import PaddleOCR

# 1. Initialize the OCR Engine ONLY for actual images
ocr_engine = PaddleOCR(use_angle_cls=True, lang='en',enable_mkldnn=False)

def extract_text_from_image(file_path):
    """Runs OCR on an image file path and returns clean text."""
    result = ocr_engine.predict(file_path)
    extracted_text = []
    if result and result[0]: 
        for line in result[0]:
            extracted_text.append(line[1][0])
    return "\n".join(extracted_text)

def process_archive(archive_folder):
    """Loops through the archive, processes all file types, and saves metadata."""
    database_chunks = []
    
    # 1. os.walk gives: the current folder (root), subfolders (dirs), and file names (files)
    for root, dirs, files in os.walk(archive_folder):
        # 2. Loop through every individual file found in that folder
        for filename in files:
            # Join root with filename to get the full, valid path
            file_path = os.path.join(root, filename)
            doc_type = "ephemera" if "ephemera" in file_path.lower() else "document"
        
        try:
            # --- 1. Handle Text-based PDFs (Novels, Codexes) - THE FAST WAY ---
            if filename.lower().endswith('.pdf'):
                print(f"Extracting text PDF: {filename}")
                doc = fitz.open(file_path)
                for page_num, page in enumerate(doc):
                    text = page.get_text("text").strip()
                    if text: # Only append if the page isn't blank
                        database_chunks.append({
                            "source": filename,
                            "page": page_num + 1,
                            "type": doc_type,
                            "content": text
                        })
                doc.close()

            # --- 2. Handle DOCX Files ---
            elif filename.lower().endswith('.docx'):
                print(f"Extracting DOCX: {filename}")
                doc = docx.Document(file_path)
                # Join all non-empty paragraphs
                text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
                if text:
                    database_chunks.append({
                        "source": filename,
                        "page": 1,
                        "type": doc_type,
                        "content": text
                    })

            # --- 3. Handle Markdown & Plain Text ---
            elif filename.lower().endswith(('.md', '.txt')):
                print(f"Extracting Text/MD: {filename}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    if text:
                        database_chunks.append({
                            "source": filename,
                            "page": 1,
                            "type": doc_type,
                            "content": text
                        })

            # --- 4. Handle Images (Scans) - THE ONLY PLACE OCR IS NEEDED ---
            elif filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                print(f"Running OCR on Image: {filename}")
                text = extract_text_from_image(file_path)
                if text:
                    database_chunks.append({
                        "source": filename,
                        "page": 1,
                        "type": doc_type,
                        "content": text
                    })
                    
        except Exception as e:
            print(f"Failed to process {filename}: {e}")
            
    return database_chunks

if __name__ == "__main__":
    TARGET_FOLDER = r"D:\projects\CODEFEST AI challenge\Ashen_Era_Archive\Ashen_Era_Archive" 
    os.makedirs(TARGET_FOLDER, exist_ok=True) 
    
    print("Starting optimized extraction pipeline...")
    extracted_data = process_archive(TARGET_FOLDER)
    
    with open("extracted_archive.json", "w", encoding="utf-8") as f:
        json.dump(extracted_data, f, indent=4)
        
    print(f"Extraction complete! Saved {len(extracted_data)} chunks to extracted_archive.json")