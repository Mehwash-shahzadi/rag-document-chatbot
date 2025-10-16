from typing import Any, List, Dict
import os
import logging
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SupportChatbot:
    """
    Main chatbot class for customer support using RAG and HuggingFace LLM.
    Uses the 'mistralai/Mistral-7B-Instruct-v0.2' model explicitly.
    """

    # OPTIMIZED PROMPT FOR CONCISE ANSWERS
    PROMPT_TEMPLATE = (
        "You are a helpful AI assistant. Answer the question based on the context provided.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Instructions:\n"
        "- Give a clear, concise answer (2-4 sentences maximum)\n"
        "- Use ONLY information from the context\n"
        "- Mention page numbers when relevant\n"
        "- If the answer isn't in the context, say 'I cannot find this information'\n\n"
        "Answer:"
    )

    def __init__(self, vector_store: Any, retriever: Any):
        """
        Args:
            vector_store: The vector store object for semantic search.
            retriever: The retriever object for context retrieval.
        """
        self.vector_store = vector_store
        self.retriever = retriever
        self.memory = ConversationBufferMemory(return_messages=True)

        # Hardcoded Mistral model ID
        self.model_id = "mistralai/Mistral-7B-Instruct-v0.2"

        # Get API token
        api_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')
        if not api_token:
            raise RuntimeError(
                "❌ HuggingFace API token not found. Please set HUGGINGFACEHUB_API_TOKEN in your .env file."
            )

        logger.info(f"Initializing HuggingFaceEndpoint with model: {self.model_id}")

        try:
            task = "text-generation"
            logger.info(f"Using task type: {task}")

            # Create the endpoint with optimized parameters
            from config.settings import Config
            llm_endpoint = HuggingFaceEndpoint(
                repo_id=self.model_id,
                task=task,
                max_new_tokens=Config.LLM_MAX_LENGTH,
                temperature=Config.LLM_TEMPERATURE,
                top_p=0.95,
                repetition_penalty=1.15,
                huggingfacehub_api_token=api_token
            )

            # Wrap with ChatHuggingFace for conversation handling
            self.llm = ChatHuggingFace(llm=llm_endpoint)
            logger.info("✅ Using ChatHuggingFace wrapper for Mistral conversational model")

        except Exception as e:
            logger.error(f"❌ Failed to initialize HuggingFace LLM: {e}")
            raise RuntimeError(f"Failed to initialize HuggingFace LLM: {e}")

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Answers a user question using retrieved context and conversation history.
        """
        try:
            logger.info(f"Processing question: {question[:50]}...")

            # Retrieve relevant context using config TOP_K
            from config.settings import Config
            results = self.retriever.retrieve(question, k=Config.TOP_K)
            logger.info(f"Retrieved {len(results)} documents")

            if not results:
                logger.warning("No relevant documents found")
                return {
                    "answer": "I couldn't find any relevant information in the knowledge base to answer your question.",
                    "confidence": 0.0,
                    "sources": "No sources found",
                    "history": self.memory.buffer
                }

            # Build context with page information
            context_parts = []
            for doc, score in results:
                page_num = doc.metadata.get("page", "unknown")
                filename = doc.metadata.get("filename", "unknown")
                context_parts.append(
                    f"[Page {page_num} - {filename}]\n{doc.page_content}"
                )
            
            context = "\n\n---\n\n".join(context_parts)
            sources = self.retriever.format_sources(results)
            confidence = self._aggregate_confidence([score for _, score in results])

            # CRITICAL FIX: Use MAX_CONTEXT_LENGTH from config
            from config.settings import Config
            context_limited = context[:Config.MAX_CONTEXT_LENGTH]
            
            prompt = self.PROMPT_TEMPLATE.format(
                context=context_limited,
                question=question
            )

            logger.info("Calling LLM with Mistral model...")
            logger.info(f"Context length: {len(context_limited)} chars")
            logger.info(f"Prompt preview: {prompt[:300]}...")

            # Call LLM
            response = self.llm.invoke(prompt)

            if hasattr(response, 'content'):
                answer = response.content
            elif isinstance(response, str):
                answer = response
            else:
                answer = str(response)

            answer = answer.strip()
            logger.info(f"✅ LLM response: {answer[:200]}...")

            if not answer or len(answer) < 5:
                logger.warning("Response too short or empty — using fallback")
                answer = self._generate_fallback_response(context, question)

            self.memory.save_context({"input": question}, {"output": answer})

            return {
                "answer": answer,
                "confidence": confidence,
                "sources": sources,
                "history": self.memory.buffer
            }

        except Exception as e:
            logger.error(f"❌ Error in ask method: {e}", exc_info=True)
            return {
                "answer": f"Sorry, I encountered an error: {str(e)}. Please try again or rephrase your question.",
                "confidence": 0.0,
                "sources": "",
                "history": self.memory.buffer,
                "error": str(e)
            }

    def _generate_fallback_response(self, context: str, question: str) -> str:
        """Generates a fallback response when LLM fails."""
        paragraphs = [p.strip() for p in context.split('\n\n') if p.strip() and len(p.strip()) > 30]

        if paragraphs:
            best_paragraph = paragraphs[0][:800]
            return (
                f"**Based on the document:**\n\n"
                f"{best_paragraph}\n\n"
                f"*Note: The AI couldn't generate a custom answer. This is direct content from your document.*"
            )

        return "I found relevant information but couldn't process it properly. Please try rephrasing your question."

    def clear_memory(self) -> None:
        """Clears the conversation memory."""
        self.memory.clear()
        logger.info("Conversation memory cleared")

    def get_conversation_history(self) -> str:
        """Returns the conversation history as a formatted string."""
        messages = self.memory.buffer
        if not messages:
            return ""
        return "\n".join(
            f"{msg['type'].capitalize()}: {msg['data']['content']}" for msg in messages
        )

    def _aggregate_confidence(self, scores: List[float]) -> float:
        """
        Aggregates similarity scores into a single confidence value (0-1).
        OPTIMIZED: Better scoring for small documents with aggressive boosting.
        """
        if not scores:
            return 0.0
        
        # Convert FAISS distance scores to confidence
        # Lower distance = higher confidence
        confidences = []
        for i, score in enumerate(scores):
            # Exponential decay for distance
            base_confidence = 1.0 / (1.0 + (score ** 0.8))
            
            # Heavy weight on top results
            weight = 1.0 / ((i + 1) ** 0.5)
            confidences.append(base_confidence * weight)
        
        # Weighted average
        total_weight = sum(1.0 / ((i + 1) ** 0.5) for i in range(len(scores)))
        weighted_avg = sum(confidences) / total_weight
        
        # Aggressive boosting for good matches
        if len(scores) >= 3:
            top_3_avg = sum(scores[:3]) / 3
            if top_3_avg < 1.5:  # Very good match
                weighted_avg = min(weighted_avg * 1.5, 0.95)
            elif top_3_avg < 2.5:  # Good match
                weighted_avg = min(weighted_avg * 1.3, 0.90)
        
        # Ensure minimum confidence for any match
        weighted_avg = max(weighted_avg, 0.50)
        
        return weighted_avg
