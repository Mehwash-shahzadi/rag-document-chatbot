# tests/test_config.py
import sys
import os

# Fix import path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config.settings import Config

print("Testing config/settings.py...")
print(f"✓ HF_TOKEN loaded: {bool(Config.HUGGINGFACEHUB_API_TOKEN)}")
print(f"✓ Embedding Model: {Config.EMBEDDING_MODEL}")
print(f"✓ LLM Model: {Config.LLM_MODEL}")
print(f"✓ Chunk Size: {Config.CHUNK_SIZE}")
print(f"✓ Chunk Overlap: {Config.CHUNK_OVERLAP}")
print(f"✓ Top K: {Config.TOP_K}")
print(f"✓ Vector DB Path: {Config.VECTOR_DB_PATH}")
print(f"✓ Raw Docs Path: {Config.RAW_DOCS_PATH}")
print("✅ Config is working!")