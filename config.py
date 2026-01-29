import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection: str = os.getenv("QDRANT_COLLECTION", "CSA")
    top_k: int = int(os.getenv("TOP_K", "6"))

    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    gemini_api_key: str = os.getenv("GOOGLE_API_KEY", "")
