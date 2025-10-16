import os
import logging
from typing import List, Optional, Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from config.settings import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStoreManager:
    """
    Manages FAISS vector store for document embeddings.

    Example:
        vsm = VectorStoreManager()
        store = vsm.create_vector_store(docs, embeddings)
        loaded = vsm.load_vector_store(embeddings)
    """

    def __init__(self):
        self.db_path = Config.VECTOR_DB_PATH
        self.index_file = os.path.join(self.db_path, "faiss_index")
        os.makedirs(self.db_path, exist_ok=True)

    def create_vector_store(self, documents: List[Document], embeddings: Embeddings) -> FAISS:
        """
        Creates a FAISS vector store from documents and embeddings, saves to disk.
        Args:
            documents (List[Document]): Documents to index.
            embeddings (Embeddings): Embeddings object (Langchain Embeddings type).
        Returns:
            FAISS: The created vector store.
        """
        try:
            logger.info(f"Creating FAISS vector store with {len(documents)} documents...")
            vector_store = FAISS.from_documents(documents, embeddings)
            vector_store.save_local(self.index_file)
            logger.info(f"Vector store saved to {self.index_file}")
            return vector_store
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}", exc_info=True)
            raise

    def load_vector_store(self, embeddings: Embeddings) -> Optional[FAISS]:
        """
        Loads FAISS vector store from disk.
        Args:
            embeddings (Embeddings): Embeddings object.
        Returns:
            Optional[FAISS]: Loaded vector store or None if not found.
        """
        if not self.exists():
            logger.warning("Vector store does not exist.")
            return None
        try:
            logger.info(f"Loading FAISS vector store from {self.index_file}...")
            # Use allow_dangerous_deserialization for newer versions of langchain
            vector_store = FAISS.load_local(
                self.index_file, 
                embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("Vector store loaded successfully")
            return vector_store
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}", exc_info=True)
            return None

    def add_documents(self, documents: List[Document], embeddings: Embeddings) -> Optional[FAISS]:
        """
        Adds new documents to the existing vector store and saves.
        Args:
            documents (List[Document]): Documents to add.
            embeddings (Embeddings): Embeddings object.
        Returns:
            Optional[FAISS]: Updated vector store or None if failed.
        """
        store = self.load_vector_store(embeddings)
        if store is None:
            logger.warning("No existing vector store. Creating new one.")
            return self.create_vector_store(documents, embeddings)
        try:
            logger.info(f"Adding {len(documents)} documents to vector store...")
            store.add_documents(documents)
            store.save_local(self.index_file)
            logger.info("Documents added and vector store updated.")
            return store
        except Exception as e:
            logger.error(f"Failed to add documents: {e}", exc_info=True)
            return None

    def delete_vector_store(self) -> None:
        """
        Deletes the FAISS vector store from disk.
        """
        try:
            if self.exists():
                for fname in os.listdir(self.db_path):
                    fpath = os.path.join(self.db_path, fname)
                    if os.path.isfile(fpath):
                        os.remove(fpath)
                logger.info("Vector store deleted.")
            else:
                logger.info("No vector store to delete.")
        except Exception as e:
            logger.error(f"Failed to delete vector store: {e}", exc_info=True)

    def exists(self) -> bool:
        """
        Checks if the FAISS vector store exists on disk.
        Returns:
            bool: True if exists, False otherwise.
        """
        faiss_file = self.index_file + ".faiss"
        pkl_file = self.index_file + ".pkl"
        exists = os.path.exists(faiss_file) and os.path.exists(pkl_file)
        logger.debug(f"Vector store exists: {exists}")
        return exists