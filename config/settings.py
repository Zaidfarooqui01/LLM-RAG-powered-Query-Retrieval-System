# config/settings.py - Enhanced with Environment Variable Support

"""
Settings and Configuration for HackRx 6.0 RAG System with Environment Support
"""

import os
from typing import Optional, Set
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    """Application settings with environment variable support"""
    
    # === CORE RAG CONFIGURATION ===
    
    # Embedding Model Configuration
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embedding_dimensions: int = int(os.getenv("EMBEDDING_DIMENSIONS", "384"))
    
    # Document Processing Configuration
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    max_chunk_size: int = int(os.getenv("MAX_CHUNK_SIZE", "2000"))
    min_chunk_size: int = int(os.getenv("MIN_CHUNK_SIZE", "100"))
    
    # Vector Store Configuration
    vector_backend: str = os.getenv("VECTOR_BACKEND", "faiss")
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    default_top_k: int = int(os.getenv("DEFAULT_TOP_K", "5"))
    
    # === FILE PROCESSING ===
    
    # File Upload Settings
    pdf_upload_path: str = os.getenv("PDF_UPLOAD_PATH", "data/uploaded_docs")
    vector_store_path: str = os.getenv("VECTOR_STORE_PATH", "data/vector_store")
    logs_path: str = os.getenv("LOGS_PATH", "data/logs")
    max_file_size_mb: int = int(os.getenv("MAX_FILE_SIZE_MB", "25"))
    
    # Supported File Types
    allowed_extensions: Set[str] = {".pdf", ".docx", ".txt"}
    supported_formats: list = ["pdf", "docx", "txt"]
    
    # === API CONFIGURATION ===
    
    # Server Configuration
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    
    # HackRx Authentication
    hackrx_token: str = os.getenv("HACKRX_TOKEN", "6ca800c46dd70bb4a8ef18a01692ac76721bb2b50303e31dbed18a186993ac1e")
    
    # Optional: OpenAI Configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    max_tokens: int = int(os.getenv("MAX_TOKENS", "1000"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.1"))
    
    # === PERFORMANCE CONFIGURATION ===
    
    max_concurrent_requests: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
    request_timeout: int = int(os.getenv("REQUEST_TIMEOUT", "60"))
    cache_ttl_hours: int = int(os.getenv("CACHE_TTL_HOURS", "24"))
    
    # System Optimization
    enable_caching: bool = os.getenv("ENABLE_CACHING", "true").lower() == "true"
    enable_explainability: bool = os.getenv("ENABLE_EXPLAINABILITY", "true").lower() == "true"
    enable_debug_logging: bool = os.getenv("ENABLE_DEBUG_LOGGING", "true").lower() == "true"
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    def __init__(self):
        """Initialize settings and create necessary directories"""
        self._create_directories()
        self._validate_settings()
        self._log_configuration()
    
    def _create_directories(self):
        """Create required directories if they don't exist"""
        directories = [
            self.pdf_upload_path,
            self.vector_store_path,
            self.logs_path
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _validate_settings(self):
        """Validate configuration settings"""
        # Validate embedding dimensions match model
        if "all-MiniLM-L6-v2" in self.embedding_model and self.embedding_dimensions != 384:
            print(f"⚠️  WARNING: Dimension mismatch! all-MiniLM-L6-v2 uses 384 dimensions, not {self.embedding_dimensions}")
            self.embedding_dimensions = 384
        
        # Validate file size
        if self.max_file_size_mb <= 0:
            raise ValueError("max_file_size_mb must be positive")
        
        # Validate chunk sizes
        if self.min_chunk_size >= self.max_chunk_size:
            raise ValueError("min_chunk_size must be less than max_chunk_size")
        
        if self.chunk_size > self.max_chunk_size:
            self.chunk_size = self.max_chunk_size
        
        if self.chunk_overlap >= self.chunk_size:
            self.chunk_overlap = self.chunk_size // 2
    
    def _log_configuration(self):
        """Log current configuration for debugging"""
        print("📊 CONFIGURATION LOADED:")
        print(f"   🤖 Embedding Model: {self.embedding_model}")
        print(f"   📏 Dimensions: {self.embedding_dimensions}")
        print(f"   📄 Chunk Size: {self.chunk_size}")
        print(f"   🔄 Chunk Overlap: {self.chunk_overlap}")
        print(f"   🗂️  Vector Store: {self.vector_store_path}")
        print(f"   🚀 API: {self.api_host}:{self.api_port}")

# Create global settings instance
settings = Settings()

# === EXPORTED CONSTANTS ===
# These maintain compatibility with your existing code

PDF_UPLOAD_PATH = settings.pdf_upload_path
VECTOR_STORE_PATH = settings.vector_store_path
LOGS_PATH = settings.logs_path
CHUNK_SIZE = settings.chunk_size
CHUNK_OVERLAP = settings.chunk_overlap
MAX_CHUNK_SIZE = settings.max_chunk_size
MIN_CHUNK_SIZE = settings.min_chunk_size
EMBEDDING_MODEL = settings.embedding_model
EMBEDDING_DIMENSIONS = settings.embedding_dimensions
VECTOR_BACKEND = settings.vector_backend
SIMILARITY_THRESHOLD = settings.similarity_threshold
DEFAULT_TOP_K = settings.default_top_k
MAX_FILE_SIZE_MB = settings.max_file_size_mb
ALLOWED_EXTENSIONS = settings.allowed_extensions
SUPPORTED_FORMATS = settings.supported_formats
HACKRX_TOKEN = settings.hackrx_token
OPENAI_API_KEY = settings.openai_api_key
