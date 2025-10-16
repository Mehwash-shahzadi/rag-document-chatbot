"""
Simple test script for the Customer Support Chatbot
Run this after processing documents to test functionality
"""

import logging
from pathlib import Path
from src.document_processor import DocumentProcessor
from src.embeddings import get_embeddings
from src.vector_store import VectorStoreManager
from src.retriever import SemanticRetriever
from src.chatbot import SupportChatbot
from config.settings import Config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main test function"""
    
    print("\n" + "="*70)
    print("🤖 CUSTOMER SUPPORT CHATBOT - TEST SCRIPT")
    print("="*70)
    
    # Validate configuration
    print("\n📋 Validating Configuration...")
    is_valid, errors = Config.validate()
    if not is_valid:
        print("❌ Configuration errors:")
        for error in errors:
            print(f"   - {error}")
        return
    
    print("✅ Configuration valid")
    print(Config.display_config())
    
    # Initialize components
    print("\n🔧 Initializing components...")
    
    try:
        # Initialize embeddings
        print("Loading embeddings model...")
        embeddings = get_embeddings()
        print("✅ Embeddings loaded")
        
        # Initialize vector store
        print("Loading vector store...")
        vector_manager = VectorStoreManager()
        vector_store = vector_manager.load_vector_store(embeddings)
        
        if vector_store is None:
            print("❌ No vector store found!")
            print("   Please upload documents first using the Streamlit app")
            print("   or run: python upload_docs.py <path_to_pdf>")
            return
        
        print("✅ Vector store loaded")
        
        # Initialize retriever
        print("Initializing retriever...")
        retriever = SemanticRetriever(vector_store)
        print("✅ Retriever initialized")
        
        # Initialize chatbot
        print("Initializing chatbot...")
        chatbot = SupportChatbot(vector_store, retriever)
        print("✅ Chatbot initialized")
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return
    
    # Test queries
    test_queries = [
        "What is this document about?",
        "Give me information from page 1",
        "What tasks are mentioned in the document?",
        "Explain the database schema requirements",
        "What are the test scenarios?"
    ]
    
    print("\n" + "="*70)
    print("🧪 RUNNING TEST QUERIES")
    print("="*70)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"Query {i}: {query}")
        print(f"{'='*70}")
        
        try:
            result = chatbot.ask(query)
            
            print(f"\n📝 Answer:")
            print(result['answer'])
            
            print(f"\n📊 Confidence: {result['confidence']:.2%}")
            
            print(f"\n📚 Sources:")
            print(result['sources'])
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("\n" + "-"*70)
    
    # Test retrieval statistics
    print("\n" + "="*70)
    print("📈 RETRIEVAL STATISTICS")
    print("="*70)
    
    test_query = "database schema"
    results = retriever.retrieve(test_query, k=5)
    stats = retriever.get_retrieval_stats(results)
    
    print(f"\nQuery: '{test_query}'")
    print(f"Documents retrieved: {stats['count']}")
    print(f"Average confidence: {stats['avg_confidence']:.2%}")
    print(f"Score range: {stats.get('min_score', 0):.4f} - {stats.get('max_score', 0):.4f}")
    print(f"Pages covered: {stats['pages_covered']}")
    print(f"Files covered: {stats['files_covered']}")
    
    # Cache statistics
    print("\n" + "="*70)
    print("💾 CACHE STATISTICS")
    print("="*70)
    
    cache_stats = embeddings.get_cache_stats()
    print(f"Cache size: {cache_stats['cache_size']} entries")
    print(f"Cache hits: {cache_stats['cache_hits']}")
    print(f"Cache misses: {cache_stats['cache_misses']}")
    print(f"Hit rate: {cache_stats['hit_rate']:.2%}")
    
    print("\n" + "="*70)
    print("✅ TEST COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()