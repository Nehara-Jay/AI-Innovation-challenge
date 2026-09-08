import os
from pathlib import Path
from functools import lru_cache
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseModel):
    # App Settings
    project_name: str = "SLIIT Codefest 2026 - Ashen Era AI Assistant"
    app_version: str = "1.0.0"
    debug: bool = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")
    api_prefix: str = "/api"
    cors_origins: List[str] = ["*"]

    # Voyage AI Embeddings
    voyage_api_key: str = os.getenv("VOYAGE_API_KEY", "")
    voyage_model: str = os.getenv("VOYAGE_MODEL", "voyage-3")

    # Qdrant Vector Database
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION_NAME", "ashen_era_corpus")
    qdrant_local_path: str = str(BASE_DIR / "data" / "local_qdrant")

    # OpenRouter / LLM Reasoning
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_base_url: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    llm_model: str = os.getenv("LLM_MODEL", "deepseek/deepseek-chat")
    temperature: float = 0.2
    max_tokens: int = 4096

    # Corpus & Data Paths
    data_dir: Path = BASE_DIR / "data"
    raw_extracted_json: Path = BASE_DIR / "data" / "extracted_archive.json"
    chunked_json: Path = BASE_DIR / "data" / "chunked_archive.json"
    local_qdrant_dir: Path = BASE_DIR / "data" / "local_qdrant"

    # Chunking Parameters
    chunk_size: int = 2000
    chunk_overlap: int = 200

    class Config:
        arbitrary_types_allowed = True


@lru_cache()
def get_settings() -> Settings:
    """Returns a cached instance of application settings."""
    return Settings()


# Default singleton instance
settings = get_settings()
