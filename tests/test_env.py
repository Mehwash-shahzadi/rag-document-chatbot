# Testing Guide - Test Each File Separately

## 🧪 Complete Testing Strategy


## **STEP 1: Test Base Configuration**

### 1.1 Test `.env` file

# test_env.py
from dotenv import load_dotenv
import os

load_dotenv()

print("Testing .env file...")
print(f"✓ HF_TOKEN exists: {bool(os.getenv('HUGGINGFACEHUB_API_TOKEN'))}")
print(f"✓ EMBEDDING_MODEL: {os.getenv('EMBEDDING_MODEL')}")
print(f"✓ LLM_MODEL: {os.getenv('LLM_MODEL')}")
print(f"✓ CHUNK_SIZE: {os.getenv('CHUNK_SIZE')}")
print("✅ .env file is working!")