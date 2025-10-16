#!/usr/bin/env python3
"""
Test script to verify HuggingFace LLM connection works with ChatHuggingFace wrapper.
"""
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# Load environment variables
load_dotenv()

def test_model(repo_id):
    """Test a specific model with ChatHuggingFace wrapper."""
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    
    print(f"\n🔄 Testing model: {repo_id}")
    
    try:
        # Step 1: Create endpoint
        llm_endpoint = HuggingFaceEndpoint(
            repo_id=repo_id,
            task="text-generation",
            max_new_tokens=256,
            temperature=0.7,
            huggingfacehub_api_token=token
        )
        print("   ✅ Endpoint created")
        
        # Step 2: Wrap with ChatHuggingFace
        model = ChatHuggingFace(llm=llm_endpoint)
        print("   ✅ ChatHuggingFace wrapper applied")
        
        # Step 3: Test invocation
        test_prompt = "What is the capital of Pakistan? Answer in one sentence."
        result = model.invoke(test_prompt)
        
        # Extract content
        response = result.content if hasattr(result, 'content') else str(result)
        
        print(f"   ✅ Success! Response:\n   {response[:200]}")
        return True, response
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False, None

def test_llm():
    """Test multiple LLM options."""
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    
    if not token:
        print("❌ Error: HUGGINGFACEHUB_API_TOKEN not found in .env file")
        return False
    
    print(f"✅ Token found: {token[:10]}...")
    
    # Test different models
    models_to_test = [
        "mistralai/Mistral-7B-Instruct-v0.2",
        "google/flan-t5-base",
        "google/flan-t5-large",
        "HuggingFaceH4/zephyr-7b-beta",
        "microsoft/DialoGPT-medium",
    ]
    
    successful_models = []
    
    for repo_id in models_to_test:
        success, response = test_model(repo_id)
        if success:
            successful_models.append(repo_id)
    
    print("\n" + "="*60)
    if successful_models:
        print("✅ Working models:")
        for model in successful_models:
            print(f"   • {model}")
        print("\nRecommended for .env file:")
        print(f"   LLM_MODEL={successful_models[0]}")
        return True
    else:
        print("❌ No models worked. Check your API token and permissions.")
        return False

if __name__ == "__main__":
    print("="*60)
    print("Testing HuggingFace LLM with ChatHuggingFace Wrapper")
    print("="*60)
    success = test_llm()
    print("="*60)
    if success:
        print("✅ All tests passed! Your configuration is working.")
    else:
        print("❌ Tests failed. Please check your configuration.")
    print("="*60)