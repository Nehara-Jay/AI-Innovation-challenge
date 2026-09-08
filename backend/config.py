import os
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseModel):
    # Voyage AI
    voyage_api_key: str = os.getenv("VOYAGE_API_KEY", "")
    voyage_model: str = os.getenv("VOYAGE_MODEL", "voyage-3")
    
    # Qdrant Vector DB
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION_NAME", "ashen_era_corpus")
    qdrant_local_path: str = str(BASE_DIR / "data" / "local_qdrant")
    
    # LLM & OpenRouter
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "deepseek/deepseek-r1:free")
    
    # Data Paths
    data_dir: Path = BASE_DIR / "data"
    raw_extracted_json: Path = BASE_DIR / "data" / "extracted_archive.json"
    chunked_json: Path = BASE_DIR / "data" / "chunked_archive.json"
    embeddings_file: Path = BASE_DIR / "data" / "embedded_archive.json"
    
    # Chunking Configuration
    chunk_size: int = 2000
    chunk_overlap: int = 200

settings = Settings()
