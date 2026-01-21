"""
Hybrid search implementation combining dense and sparse vectors with RRF.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import QueryRequest, QueryRequestBatch, SearchRequest, Prefetch, SparseVector
from langchain_core.documents import Document

from sparse_utils import BM25Encoder

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid retriever combining dense and sparse search with RRF."""

    def __init__(
        self,
        qdrant_client: QdrantClient,
        dense_model: SentenceTransformer,
        sparse_encoder: BM25Encoder,
        collection_name: str = "simple_rag",
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        top_k: int = 5,
        rrf_k: int = 60
    ):
        self.client = qdrant_client
        self.dense_model = dense_model
        self.sparse_encoder = sparse_encoder
        self.collection_name = collection_name
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.top_k = top_k
        self.rrf_k = rrf_k

    def _dense_search(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Perform dense vector search."""
        query_vector = self.dense_model.encode(query).tolist()

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using="dense" if 'hybrid' in self.collection_name else None,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )

        return [
            {
                'id': hit.id,
                'score': hit.score,
                'payload': hit.payload,
                'search_type': 'dense'
            }
            for hit in results.points
        ]

    def _sparse_search(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Perform sparse vector search."""
        try:
            sparse_vector = self.sparse_encoder.encode(query)
        except ValueError:
            logger.warning("BM25 model not fitted, skipping sparse search")
            return []

        if not sparse_vector:
            logger.warning("Empty sparse vector for query, returning empty results")
            return []

        # Convert to Qdrant sparse vector format
        indices = list(sparse_vector.keys())
        values = list(sparse_vector.values())
        qdrant_sparse = SparseVector(indices=indices, values=values)

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=qdrant_sparse,
            using="sparse" if 'hybrid' in self.collection_name else None,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )

        return [
            {
                'id': hit.id,
                'score': hit.score,
                'payload': hit.payload,
                'search_type': 'sparse'
            }
            for hit in results.points
        ]

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """Combine results using Reciprocal Rank Fusion."""
        # Create document score mapping
        doc_scores = {}

        # Process dense results
        for rank, result in enumerate(dense_results, 1):
            doc_id = result['id']
            rrf_score = self.dense_weight / (k + rank)
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {'score': 0, 'results': []}
            doc_scores[doc_id]['score'] += rrf_score
            doc_scores[doc_id]['results'].append(result)

        # Process sparse results
        for rank, result in enumerate(sparse_results, 1):
            doc_id = result['id']
            rrf_score = self.sparse_weight / (k + rank)
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {'score': 0, 'results': []}
            doc_scores[doc_id]['score'] += rrf_score
            doc_scores[doc_id]['results'].append(result)

        # Sort by combined RRF score and return top results
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1]['score'], reverse=True)

        combined_results = []
        for doc_id, data in sorted_docs[:self.top_k]:
            # Use the result with the highest individual score for payload
            best_result = max(data['results'], key=lambda x: x.get('score', 0))
            combined_results.append({
                'id': doc_id,
                'score': data['score'],
                'payload': best_result['payload'],
                'search_type': 'hybrid'
            })

        return combined_results

    def search(self, query: str) -> List[Document]:
        """Perform hybrid search and return documents."""
        try:
            # Get results from both dense and sparse search
            dense_results = self._dense_search(query, limit=20)
            sparse_results = self._sparse_search(query, limit=20)

            # Combine using RRF
            combined_results = self._reciprocal_rank_fusion(
                dense_results, sparse_results, k=self.rrf_k
            )

            # Convert to Document objects
            documents = [
                Document(
                    page_content=result['payload']['text'],
                    metadata={
                        'id': result['id'],
                        'score': result['score'],
                        'search_type': result['search_type']
                    }
                )
                for result in combined_results
            ]

            logger.debug(f"Hybrid search returned {len(documents)} results for query: {query[:50]}...")
            return documents

        except Exception as e:
            logger.error(f"Hybrid search error: {e}")
            # Fallback to dense search
            logger.info("Falling back to dense search")
            dense_results = self._dense_search(query, limit=self.top_k)
            return [
                Document(
                    page_content=result['payload']['text'],
                    metadata={
                        'id': result['id'],
                        'score': result['score'],
                        'search_type': 'dense_fallback'
                    }
                )
                for result in dense_results
            ]

    def search_dense_only(self, query: str) -> List[Document]:
        """Fallback method using dense search only."""
        results = self._dense_search(query, limit=self.top_k)
        return [
            Document(
                page_content=result['payload']['text'],
                metadata={
                    'id': result['id'],
                    'score': result['score'],
                    'search_type': 'dense'
                }
            )
            for result in results
        ]
