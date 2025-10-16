"""
Document Upload Script
Upload and process documents to build the vector store
Usage: python upload_docs.py <path_to_document>
"""

import sys
import logging
from pathlib import Path
from src.document_processor import DocumentProcessor
from src.embeddings import get_embeddings
from src.vector_store import VectorStoreManager
from config.settings import Config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def upload_documents(file_paths: list):
    """
    Upload and process documents to the vector store.
    
    Args:
        file_paths: List of paths to documents
    """
    print("\n" + "="*70)
    print("📁 DOCUMENT UPLOAD & PROCESSING")
    print("="*70)
    
    # Validate configuration
    print("\n1️⃣ Validating Configuration...")
    is_valid, errors = Config.validate()
    if not is_valid:
        print("❌ Configuration errors:")
        for error in errors:
            print(f"   - {error}")
        return False
    print("✅ Configuration valid")
    
    # Validate files
    print("\n2️⃣ Validating Files...")
    valid_files = []
    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            print(f"❌ File not found: {file_path}")
            continue
        if not path.is_file():
            print(f"❌ Not a file: {file_path}")
            continue
        if path.suffix.lower() not in ['.pdf', '.txt']:
            print(f"❌ Unsupported file type: {file_path}")
            continue
        valid_files.append(str(path.absolute()))
        print(f"✅ {path.name}")
    
    if not valid_files:
        print("❌ No valid files to process")
        return False
    
    print(f"\n📊 Total files to process: {len(valid_files)}")
    
    # Process documents
    print("\n3️⃣ Processing Documents...")
    try:
        processor = DocumentProcessor(valid_files)
        documents = processor.process()
        
        if not documents:
            print("❌ No documents were processed")
            return False
        
        print(f"✅ Processed {len(documents)} document chunks")
        
        # Show processing stats
        stats = processor.get_processing_stats(documents)
        print(f"\n📈 Processing Statistics:")
        print(f"   Total chunks: {stats['total_chunks']}")
        print(f"   Total characters: {stats['total_chars']:,}")
        print(f"   Average chunk size: {stats['avg_chunk_size']:.0f} chars")
        print(f"   Files: {', '.join(stats['files'])}")
        print(f"   Pages: {len(stats['pages'])} pages")
        
    except Exception as e:
        print(f"❌ Error processing documents: {e}")
        logger.error("Document processing failed", exc_info=True)
        return False
    
    # Initialize embeddings
    print("\n4️⃣ Loading Embeddings Model...")
    try:
        embeddings = get_embeddings()
        print(f"✅ Loaded model: {Config.EMBEDDING_MODEL}")
    except Exception as e:
        print(f"❌ Error loading embeddings: {e}")
        return False
    
    # Create/update vector store
    print("\n5️⃣ Building Vector Store...")
    try:
        vector_manager = VectorStoreManager()
        
        # Check if vector store exists
        if vector_manager.exists():
            print("⚠️  Existing vector store found")
            response = input("Do you want to (A)ppend or (R)eplace? [A/R]: ").strip().upper()
            
            if response == 'R':
                print("🗑️  Deleting existing vector store...")
                vector_manager.delete_vector_store()
                print("Creating new vector store...")
                vector_store = vector_manager.create_vector_store(documents, embeddings)
            else:
                print("➕ Adding documents to existing vector store...")
                vector_store = vector_manager.add_documents(documents, embeddings)
        else:
            print("Creating new vector store...")
            vector_store = vector_manager.create_vector_store(documents, embeddings)
        
        print(f"✅ Vector store ready at: {Config.VECTOR_DB_PATH}")
        
    except Exception as e:
        print(f"❌ Error creating vector store: {e}")
        logger.error("Vector store creation failed", exc_info=True)
        return False
    
    # Test retrieval
    print("\n6️⃣ Testing Retrieval...")
    try:
        from src.retriever import SemanticRetriever
        retriever = SemanticRetriever(vector_store)
        
        test_query = "What is this document about?"
        results = retriever.retrieve(test_query, k=3)
        
        print(f"Test query: '{test_query}'")
        print(f"Retrieved {len(results)} results")
        
        if results:
            top_doc, top_score = results[0]
            print(f"\nTop result:")
            print(f"  Page: {top_doc.metadata.get('page', 'N/A')}")
            print(f"  File: {top_doc.metadata.get('filename', 'unknown')}")
            print(f"  Score: {top_score:.4f}")
            print(f"  Preview: {top_doc.page_content[:150]}...")
        
        print("✅ Retrieval test passed")
        
    except Exception as e:
        print(f"⚠️  Retrieval test failed: {e}")
    
    # Success summary
    print("\n" + "="*70)
    print("✅ UPLOAD COMPLETE!")
    print("="*70)
    print(f"\n📚 Documents processed: {len(valid_files)}")
    print(f"📦 Chunks created: {len(documents)}")
    print(f"💾 Vector store location: {Config.VECTOR_DB_PATH}")
    print(f"\n🚀 You can now:")
    print(f"   - Run: streamlit run app.py")
    print(f"   - Or test: python test_chatbot.py")
    print("="*70 + "\n")
    
    return True

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python upload_docs.py <path_to_document> [additional_documents...]")
        print("\nExamples:")
        print('  python upload_docs.py "docs\\Backend Developer Assessment.pdf"')
        print('  python upload_docs.py "C:\\Users\\LAPTOP POINT\\Desktop\\customer-support-chatbot\\docs\\Backend Developer Assessment.pdf"')
        print('  python upload_docs.py "docs\\file1.pdf" "docs\\file2.pdf"')
        print("\n💡 Tip: Use quotes around paths with spaces!")
        sys.exit(1)
    
    file_paths = sys.argv[1:]
    success = upload_documents(file_paths)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()