from typing import List, Dict, Any
import os
import logging

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config.settings import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_file_extension(file_path: str) -> str:
    """
    Returns the file extension in lowercase (without dot).
    Args:
        file_path (str): Path to the file.
    Returns:
        str: File extension (e.g., 'pdf', 'txt').
    """
    return os.path.splitext(file_path)[1][1:].lower()

class DocumentProcessor:
    """
    Handles loading and splitting of documents (PDF, TXT) for RAG chatbot.
    Improved version with better metadata handling and chunking strategy.
    """

    def __init__(self, file_paths: List[str]):
        """
        Initializes the processor with a list of file paths.
        Args:
            file_paths (List[str]): List of document file paths.
        """
        self.file_paths = file_paths
        logger.info(f"Initializing DocumentProcessor with {len(file_paths)} files")

    def load_documents(self) -> List[Document]:
        """
        Loads documents using appropriate loaders based on file extension.
        Returns:
            List[Document]: List of loaded Document objects.
        Raises:
            ValueError: If file format is unsupported or file is empty.
        """
        documents = []
        for file_path in self.file_paths:
            ext = get_file_extension(file_path)
            filename = os.path.basename(file_path)
            
            try:
                logger.info(f"Loading file: {filename} (type: {ext})")
                
                if ext == "pdf":
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    logger.info(f"✅ Loaded {len(docs)} pages from {filename}")
                    
                elif ext == "txt":
                    loader = TextLoader(file_path, encoding="utf-8")
                    docs = loader.load()
                    logger.info(f"✅ Loaded text file {filename}")
                    
                else:
                    raise ValueError(f"Unsupported file format: {ext} ({file_path})")
                
                if not docs:
                    raise ValueError(f"File is empty or unreadable: {file_path}")
                
                # Enhance metadata with original filename
                for doc in docs:
                    doc.metadata["original_filename"] = filename
                    
                documents.extend(docs)
                
            except Exception as e:
                logger.error(f"❌ Error loading {file_path}: {e}")
                raise
                
        logger.info(f"Total documents loaded: {len(documents)}")
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits documents into chunks using RecursiveCharacterTextSplitter.
        Preserves metadata: filename, page number, chunk index.
        
        IMPROVED: Better separators and metadata handling for page-aware chunking.
        
        Args:
            documents (List[Document]): List of loaded Document objects.
        Returns:
            List[Document]: List of split Document objects with enhanced metadata.
        """
        # Better separators for cleaner chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=[
                "\n\n",      # Paragraph breaks (highest priority)
                "\n",        # Line breaks
                ". ",        # Sentence ends
                "! ",        # Exclamation ends
                "? ",        # Question ends
                "; ",        # Semicolon
                ", ",        # Comma
                " ",         # Space
                ""           # Character-level (last resort)
            ],
            length_function=len,
            keep_separator=True  # Keep separators for better context
        )
        
        split_docs = []
        total_chunks = 0
        
        for doc in documents:
            # Extract metadata
            filename = doc.metadata.get("original_filename") or doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page")
            source_path = doc.metadata.get("source", "")
            
            # Split the document content
            chunks = splitter.split_text(doc.page_content)
            
            logger.info(f"Split {filename} (page {page}) into {len(chunks)} chunks")
            
            for idx, chunk in enumerate(chunks):
                # Create enhanced metadata
                metadata = {
                    "filename": filename,
                    "source": source_path,
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk)
                }
                
                # Add page number (1-indexed for user-friendliness)
                if page is not None:
                    metadata["page"] = page + 1
                else:
                    metadata["page"] = "N/A"
                
                # Create new document with chunk and metadata
                split_docs.append(
                    Document(
                        page_content=chunk.strip(),
                        metadata=metadata
                    )
                )
                total_chunks += 1
        
        logger.info(f"✅ Total chunks created: {total_chunks}")
        return split_docs

    def process(self) -> List[Document]:
        """
        Loads and splits documents, returning processed chunks with metadata.
        Returns:
            List[Document]: List of processed Document objects.
        """
        logger.info("Starting document processing...")
        
        # Load documents
        docs = self.load_documents()
        if not docs:
            logger.warning("No documents loaded.")
            return []
        
        # Split into chunks
        split_docs = self.split_documents(docs)
        
        logger.info(f"✅ Document processing complete: {len(split_docs)} chunks ready")
        return split_docs

    def get_processing_stats(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Returns statistics about processed documents.
        
        Args:
            documents (List[Document]): Processed documents
            
        Returns:
            Dict containing processing statistics
        """
        if not documents:
            return {"total_chunks": 0}
        
        stats = {
            "total_chunks": len(documents),
            "total_chars": sum(len(doc.page_content) for doc in documents),
            "avg_chunk_size": sum(len(doc.page_content) for doc in documents) / len(documents),
            "files": list(set(doc.metadata.get("filename", "unknown") for doc in documents)),
            "pages": list(set(doc.metadata.get("page", "N/A") for doc in documents))
        }
        
        return stats