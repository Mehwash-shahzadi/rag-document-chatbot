from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from config.settings import Config
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HuggingFaceEmbeddingsWrapper(Embeddings):
    """
    Wrapper for HuggingFaceEmbeddings with caching and batch processing.
    Properly inherits from Embeddings to avoid FAISS warnings.

    Example:
        embeddings = HuggingFaceEmbeddingsWrapper()
        vec = embeddings.embed_query("Hello world")
        vecs = embeddings.embed_documents(["doc1", "doc2"])
    """

    def __init__(self):
        try:
            self.model_name = Config.EMBEDDING_MODEL
            logger.info(f"Loading embedding model: {self.model_name}")
            
            # Initialize HuggingFaceEmbeddings without API token (runs locally)
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={'device': 'cpu'},  # Explicitly use CPU
                encode_kwargs={'normalize_embeddings': True}  # Normalize for better similarity
            )
            
            logger.info("✅ Embedding model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load HuggingFace embeddings model: {e}")
            raise RuntimeError(f"Failed to load HuggingFace embeddings model: {e}")
        
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single text string with caching.
        
        Args:
            text (str): Input text.
        Returns:
            List[float]: Embedding vector.
        """
        # Check cache first
        if text in self._cache:
            self._cache_hits += 1
            return self._cache[text]
        
        self._cache_misses += 1
        
        try:
            vec = self.embeddings.embed_query(text)
            self._cache[text] = vec
            return vec
        except Exception as e:
            logger.error(f"❌ Error embedding query: {e}")
            raise

    def embed_documents(self, texts: List[str], show_progress: bool = True) -> List[List[float]]:
        """
        Embed a list of text strings with optional progress bar and caching.
        
        Args:
            texts (List[str]): List of input texts.
            show_progress (bool): Whether to show a progress bar.
        Returns:
            List[List[float]]: List of embedding vectors.
        """
        results = []
        cached_count = 0
        
        # Determine whether to show progress bar
        iterator = tqdm(texts, desc="Embedding documents", unit="doc") if show_progress and len(texts) > 10 else texts
        
        for text in iterator:
            if text in self._cache:
                results.append(self._cache[text])
                cached_count += 1
            else:
                try:
                    vec = self.embeddings.embed_query(text)
                    self._cache[text] = vec
                    results.append(vec)
                except Exception as e:
                    logger.error(f"❌ Error embedding document: {e}")
                    # Use zero vector as fallback
                    results.append([0.0] * 384)  # all-MiniLM-L6-v2 has 384 dimensions
        
        if cached_count > 0:
            logger.info(f"Cache hit rate: {cached_count}/{len(texts)} ({cached_count/len(texts)*100:.1f}%)")
        
        return results
    
    async def aembed_query(self, text: str) -> List[float]:
        """Async version of embed_query (calls sync version)."""
        return self.embed_query(text)
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async version of embed_documents (calls sync version)."""
        return self.embed_documents(texts, show_progress=False)

    def get_cache_stats(self) -> dict:
        """
        Returns cache statistics.
        
        Returns:
            Dictionary with cache performance metrics
        """
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            "cache_size": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
            "total_requests": total_requests
        }

    def clear_cache(self):
        """Clears the embedding cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("Embedding cache cleared")

def get_embeddings() -> HuggingFaceEmbeddingsWrapper:
    """
    Factory function to get an initialized HuggingFaceEmbeddingsWrapper.
    
    Returns:
        HuggingFaceEmbeddingsWrapper: The embeddings object.
    """
    return HuggingFaceEmbeddingsWrapper()