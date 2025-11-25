# config.py
import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Config:
    # Gemini API
    GEMINI_API_KEY: str = "AIzaSyCB3FiZr7Nb4YPUjxW1Y56uQwhuoZ-O1EM"
    GEMINI_MODEL: str = "gemini‑2.5‑flash"
    EMBEDDING_MODEL: str = "models/embedding-001"
    
    # Vector Database
    VECTOR_DB_TYPE: str = "chromadb"  # or "pinecone", "weaviate"
    CHROMA_DB_PATH: str = "./data/chroma_db"
    
    # Document Processing
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_DOCUMENT_SIZE: int = 50 * 1024 * 1024  # 50MB
    
    # Memory Settings
    SHORT_TERM_MEMORY_SIZE: int = 20
    SESSION_TIMEOUT: int = 3600  # 1 hour
    
    # Learning Settings
    FEEDBACK_STORAGE_PATH: str = "./data/feedback"
    MODEL_CACHE_PATH: str = "./data/models"
    
    # Performance
    CACHE_TTL: int = 3600  # 1 hour
    MAX_CONCURRENT_REQUESTS: int = 10

config = Config()