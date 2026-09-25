import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LiveKit Configuration
    livekit_url: str
    livekit_api_key: str
    livekit_api_secret: str

    # LLM Configuration
    groq_api_key: str
    groq_model: str = "openai/gpt-oss-120b"

    # Document Ingestion
    docs_dir: str = "docs"

    # Embedding & Vector Store
    embedding_model: str = "all-MiniLM-L6-v2"
    qdrant_path: str = "./data/qdrant"
    collection_name: str = "telecom_docs"
    chunk_size: int = 500

    # Voice Configuration
    stt_model: str = "deepgram/nova-3"
    stt_language: str = "multi"
    tts_model: str = "inworld/inworld-tts-2"
    tts_voice: str = "Ashley"


settings = Settings()

# Set environment variables for LiveKit
os.environ.setdefault("LIVEKIT_URL", settings.livekit_url)
os.environ.setdefault("LIVEKIT_API_KEY", settings.livekit_api_key)
os.environ.setdefault("LIVEKIT_API_SECRET", settings.livekit_api_secret)
os.environ.setdefault("GROQ_API_KEY", settings.groq_api_key)
