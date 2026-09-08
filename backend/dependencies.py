"""
FastAPI dependency injection providers for database connections,
AI clients (Voyage AI, OpenRouter/LLM), and shared application services.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

import voyageai
from openai import OpenAI
from qdrant_client import QdrantClient
from fastapi import Depends

from backend.config import Settings, get_settings


@lru_cache()
def get_qdrant() -> QdrantClient:
    """
    Creates and caches a singleton Qdrant client connection (Docker/remote or local embedded).
    """
    settings = get_settings()
    url = settings.qdrant_url
    if url == "local" or not url.startswith("http"):
        local_path = Path(settings.qdrant_local_path)
        local_path.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(local_path))
    else:
        return QdrantClient(
            url=url,
            api_key=settings.qdrant_api_key if settings.qdrant_api_key else None,
        )


@lru_cache()
def get_voyage() -> voyageai.Client:
    """
    Creates and caches singleton Voyage AI client instance.
    """
    settings = get_settings()
    if not settings.voyage_api_key:
        raise ValueError("VOYAGE_API_KEY is not configured in .env")
    return voyageai.Client(api_key=settings.voyage_api_key)


@lru_cache()
def get_llm_client() -> OpenAI:
    """
    Creates and caches singleton OpenAI-compatible client for OpenRouter / DeepSeek.
    """
    settings = get_settings()
    if not settings.openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY is not configured in .env")
    return OpenAI(
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
    )
