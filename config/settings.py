import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """
    Configuration class for the Customer Support Chatbot.
    All settings are loaded from environment variables with sensible defaults.
    """
    
    # ===== API Tokens =====
    HUGGINGFACE_API_TOKEN: Optional[str] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    
    # ===== Model Settings =====
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", 
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    LLM_MODEL: str = os.getenv(
        "LLM_MODEL", 
        "mistralai/Mistral-7B-Instruct-v0.2"
    )
    
    # ===== Vector Store Settings =====
    VECTOR_DB_PATH: str = os.getenv("VECTOR_DB_PATH", "./vector_db")
    
    # ===== Document Processing Settings (OPTIMIZED FOR BETTER RESULTS) =====
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1500"))  # Increased from 1000
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "300"))  # Increased from 200
    
    # ===== Retrieval Settings (OPTIMIZED) =====
    TOP_K: int = int(os.getenv("TOP_K", "10"))  # Increased from 5 for better coverage
    
    # ===== File Upload Settings =====
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
    ALLOWED_FILE_TYPES: list = ["pdf", "txt"]
    
    # ===== LLM Settings (OPTIMIZED) =====
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))  # Lower for accuracy
    LLM_MAX_LENGTH: int = int(os.getenv("LLM_MAX_LENGTH", "768"))  # Increased for longer answers
    
    # ===== Context Settings =====
    MAX_CONTEXT_LENGTH: int = int(os.getenv("MAX_CONTEXT_LENGTH", "6000"))  # NEW: Control context size
    
    @classmethod
    def validate(cls) -> tuple[bool, list[str]]:
        """
        Validates that all required configuration values are set.
        
        Returns:
            tuple[bool, list[str]]: (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required fields
        if not cls.HUGGINGFACE_API_TOKEN:
            errors.append(
                "HUGGINGFACEHUB_API_TOKEN is not set. "
                "Please add it to your .env file."
            )
        
        # Validate numeric ranges
        if cls.CHUNK_SIZE < 100 or cls.CHUNK_SIZE > 5000:
            errors.append(f"CHUNK_SIZE must be between 100 and 5000 (current: {cls.CHUNK_SIZE})")
        
        if cls.CHUNK_OVERLAP >= cls.CHUNK_SIZE:
            errors.append(
                f"CHUNK_OVERLAP ({cls.CHUNK_OVERLAP}) must be less than "
                f"CHUNK_SIZE ({cls.CHUNK_SIZE})"
            )
        
        if cls.TOP_K < 1 or cls.TOP_K > 20:
            errors.append(f"TOP_K must be between 1 and 20 (current: {cls.TOP_K})")
        
        return len(errors) == 0, errors
    
    @classmethod
    def display_config(cls) -> str:
        """
        Returns a formatted string of current configuration (hiding sensitive data).
        """
        token_display = "***" + cls.HUGGINGFACE_API_TOKEN[-4:] if cls.HUGGINGFACE_API_TOKEN else "NOT SET"
        
        return f"""
Configuration Settings:
=======================
API Token: {token_display}
Embedding Model: {cls.EMBEDDING_MODEL}
LLM Model: {cls.LLM_MODEL}
Vector DB Path: {cls.VECTOR_DB_PATH}
Chunk Size: {cls.CHUNK_SIZE}
Chunk Overlap: {cls.CHUNK_OVERLAP}
Top K Results: {cls.TOP_K}
Max File Size: {cls.MAX_FILE_SIZE_MB} MB
LLM Temperature: {cls.LLM_TEMPERATURE}
LLM Max Length: {cls.LLM_MAX_LENGTH}
Max Context Length: {cls.MAX_CONTEXT_LENGTH}
"""