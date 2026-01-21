"""
Cross-encoder reranker for improving retrieval quality.
"""
import logging
from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Cross-encoder reranker for improving document ranking."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        max_length: int = 512
    ):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.model: Optional[CrossEncoder] = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the cross-encoder model."""
        try:
            self.model = CrossEncoder(
                self.model_name,
                device=self.device,
                max_length=self.max_length
            )
            logger.info(f"Loaded cross-encoder model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder model: {e}")
            raise

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None
    ) -> List[Document]:
        """
        Rerank documents based on query-document relevance.

        Args:
            query: The search query
            documents: List of documents to rerank
            top_k: Number of top documents to return (None = all)

        Returns:
            Reranked list of documents with updated scores
        """
        if not documents:
            return []

        if self.model is None:
            logger.warning("Cross-encoder model not loaded, returning original documents")
            return documents

        try:
            # Prepare query-document pairs for cross-encoder
            query_doc_pairs = [
                [query, doc.page_content] for doc in documents
            ]

            # Get cross-encoder scores (higher = more relevant)
            scores = self.model.predict(query_doc_pairs)

            # Update documents with new scores
            reranked_docs = []
            for doc, score in zip(documents, scores):
                # Update metadata with reranking info
                new_metadata = doc.metadata.copy()
                new_metadata.update({
                    'rerank_score': float(score),
                    'original_score': doc.metadata.get('score', 0),
                    'reranked': True
                })

                reranked_doc = Document(
                    page_content=doc.page_content,
                    metadata=new_metadata
                )
                reranked_docs.append((reranked_doc, score))

            # Sort by cross-encoder score (descending)
            reranked_docs.sort(key=lambda x: x[1], reverse=True)

            # Return top-k if specified
            result_docs = [doc for doc, _ in reranked_docs]
            if top_k is not None:
                result_docs = result_docs[:top_k]

            logger.debug(f"Reranked {len(documents)} documents to {len(result_docs)} for query: {query[:50]}...")
            return result_docs

        except Exception as e:
            logger.error(f"Reranking error: {e}")
            logger.info("Returning original documents due to reranking error")
            return documents

    def rerank_batch(
        self,
        queries: List[str],
        document_lists: List[List[Document]],
        top_k: Optional[int] = None
    ) -> List[List[Document]]:
        """
        Batch rerank multiple queries.

        Args:
            queries: List of queries
            document_lists: List of document lists (one per query)
            top_k: Number of top documents per query

        Returns:
            List of reranked document lists
        """
        if len(queries) != len(document_lists):
            raise ValueError("Number of queries must match number of document lists")

        results = []
        for query, docs in zip(queries, document_lists):
            reranked = self.rerank(query, docs, top_k=top_k)
            results.append(reranked)

        return results


def create_reranker(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device: str = "cpu"
) -> CrossEncoderReranker:
    """Factory function to create a reranker instance."""
