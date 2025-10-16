from typing import List, Tuple, Any, Optional
from config.settings import Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticRetriever:
    """
    SemanticRetriever enables semantic search over a vector store.
    IMPROVED: Better scoring, filtering, and context formatting.

    Example:
        retriever = SemanticRetriever(vector_store)
        results = retriever.retrieve("What is your refund policy?")
        print(retriever.retrieve_with_context("What is your refund policy?"))
    """

    def __init__(self, vector_store: Any):
        """
        Args:
            vector_store: A vector store object supporting similarity_search_with_score().
        """
        self.vector_store = vector_store
        logger.info("SemanticRetriever initialized")

    def retrieve(self, query: str, k: int = None, score_threshold: float = None) -> List[Tuple[Any, float]]:
        """
        Performs semantic similarity search for the query.

        Args:
            query (str): The search query.
            k (int, optional): Number of top results to return. Defaults to Config.TOP_K.
            score_threshold (float, optional): Filter results by minimum score.

        Returns:
            List[Tuple[Document, float]]: List of (document, similarity_score) tuples, sorted by relevance.

        Score interpretation:
            Lower scores mean higher similarity (distance-based). Use for ranking/citations.
        """
        if k is None:
            k = Config.TOP_K
        
        try:
            logger.info(f"Retrieving top {k} documents for query: '{query[:50]}...'")
            
            # Get results from vector store
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            # Filter by score threshold if provided
            if score_threshold is not None:
                results = [(doc, score) for doc, score in results if score <= score_threshold]
                logger.info(f"After score filtering: {len(results)} documents")
            
            # Sort by ascending score (most relevant first - lower score = more similar)
            results.sort(key=lambda x: x[1])
            
            # Log top results
            if results:
                logger.info(f"Top result score: {results[0][1]:.4f}")
                logger.info(f"Top result page: {results[0][0].metadata.get('page', 'N/A')}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []

    def retrieve_with_context(self, query: str, k: int = None, include_scores: bool = True) -> str:
        """
        Retrieves documents and formats them with metadata and confidence.

        Args:
            query (str): The search query.
            k (int, optional): Number of top results to return.
            include_scores (bool): Whether to include confidence scores.

        Returns:
            str: Formatted string with sources, scores, and context.
        """
        results = self.retrieve(query, k)
        
        if not results:
            return "No relevant documents found."
        
        formatted = []
        for doc, score in results:
            confidence = self._score_to_confidence(score)
            meta = doc.metadata
            
            # Format metadata info
            page_info = f"Page {meta.get('page', 'N/A')}"
            chunk_info = f"Chunk {meta.get('chunk_index', 0) + 1}/{meta.get('total_chunks', 1)}"
            filename = meta.get('filename', 'unknown')
            
            # Build formatted section
            header = f"📄 [{filename}] - {page_info} ({chunk_info})"
            
            if include_scores:
                header += f"\n   Confidence: {confidence:.2%} | Score: {score:.4f}"
            
            formatted.append(
                f"{header}\n"
                f"{'-' * 70}\n"
                f"{doc.page_content}\n"
            )
        
        return "\n".join(formatted)

    def format_sources(self, results: List[Tuple[Any, float]]) -> str:
        """
        Formats a list of results into citation text.
        IMPROVED: Better formatting with page numbers and confidence.

        Args:
            results (List[Tuple[Document, float]]): Results from retrieve().

        Returns:
            str: Citation string.
        """
        if not results:
            return "No sources"
        
        citations = []
        for idx, (doc, score) in enumerate(results, 1):
            meta = doc.metadata
            confidence = self._score_to_confidence(score)
            
            citation = (
                f"{idx}. {meta.get('filename', 'unknown')} "
                f"(Page {meta.get('page', 'N/A')}, "
                f"Confidence: {confidence:.1%})"
            )
            citations.append(citation)
        
        return "\n".join(citations)

    def get_page_specific_context(self, query: str, page_number: int, k: int = 5) -> List[Tuple[Any, float]]:
        """
        Retrieves documents from a specific page.
        
        Args:
            query (str): Search query
            page_number (int): Specific page to search
            k (int): Number of results
            
        Returns:
            List of (document, score) tuples from specified page
        """
        # Get more results initially
        all_results = self.retrieve(query, k=k*3)
        
        # Filter for specific page
        page_results = [
            (doc, score) for doc, score in all_results 
            if doc.metadata.get('page') == page_number
        ]
        
        # Return top k from that page
        return page_results[:k]

    def get_multi_page_context(self, query: str, page_range: Tuple[int, int], k: int = 10) -> List[Tuple[Any, float]]:
        """
        Retrieves documents from a range of pages.
        
        Args:
            query (str): Search query
            page_range (Tuple[int, int]): (start_page, end_page) inclusive
            k (int): Number of results
            
        Returns:
            List of (document, score) tuples from page range
        """
        start_page, end_page = page_range
        
        # Get more results initially
        all_results = self.retrieve(query, k=k*2)
        
        # Filter for page range
        range_results = [
            (doc, score) for doc, score in all_results 
            if isinstance(doc.metadata.get('page'), int) and 
               start_page <= doc.metadata.get('page') <= end_page
        ]
        
        # Sort by page number, then by score
        range_results.sort(key=lambda x: (x[0].metadata.get('page', 999), x[1]))
        
        return range_results[:k]

    def _score_to_confidence(self, score: float) -> float:
        """
        Converts a similarity score to a confidence value (0-1).
        Lower score = higher confidence (FAISS distance metric).

        Args:
            score (float): Similarity score (distance).

        Returns:
            float: Confidence value (1.0 = most confident).
        """
        # For L2 distance: confidence = 1 / (1 + score)
        # This maps [0, inf) -> (0, 1]
        confidence = 1.0 / (1.0 + score)
        return confidence

    def get_retrieval_stats(self, results: List[Tuple[Any, float]]) -> dict:
        """
        Returns statistics about retrieval results.
        
        Args:
            results: List of (document, score) tuples
            
        Returns:
            Dictionary with retrieval statistics
        """
        if not results:
            return {
                "count": 0,
                "avg_confidence": 0.0,
                "pages_covered": []
            }
        
        confidences = [self._score_to_confidence(score) for _, score in results]
        pages = [doc.metadata.get('page') for doc, _ in results]
        
        return {
            "count": len(results),
            "avg_confidence": sum(confidences) / len(confidences),
            "min_score": min(score for _, score in results),
            "max_score": max(score for _, score in results),
            "pages_covered": sorted(set(p for p in pages if p != 'N/A')),
            "files_covered": list(set(doc.metadata.get('filename') for doc, _ in results))
        }