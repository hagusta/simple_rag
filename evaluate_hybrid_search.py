"""
Evaluation script to compare dense vs hybrid search with reranking.
"""
import pickle
import logging
import time
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from langchain_core.documents import Document

from hybrid_search import HybridRetriever
from reranker import create_reranker
from sparse_utils import BM25Encoder

logger = logging.getLogger(__name__)


def load_test_data() -> List[Dict[str, Any]]:
    """Load evaluation queries and relevant documents."""
    try:
        with open('data/queries.pkl', 'rb') as f:
            queries = pickle.load(f)
        with open('data/relevant_docs.pkl', 'rb') as f:
            relevant_docs = pickle.load(f)

        # Combine queries with relevant docs
        test_data = []
        for i, query in enumerate(queries):
            relevant_doc_ids = relevant_docs.get(i, [])
            test_data.append({
                'query': str(query),  # Ensure query is a string
                'relevant_doc_ids': relevant_doc_ids,  # Keep as list for now
                'query_id': i
            })

        logger.info(f"Loaded {len(test_data)} test queries")
        return test_data[:50]  # Limit for evaluation

    except FileNotFoundError:
        logger.warning("Evaluation data not found, creating sample queries")
        return [
            {'query': 'who is socrates?', 'relevant_doc_ids': set(), 'query_id': 0},
            {'query': 'what is justice?', 'relevant_doc_ids': set(), 'query_id': 1},
            {'query': 'philosopher king', 'relevant_doc_ids': set(), 'query_id': 2},
        ]


def evaluate_search_method(
    search_function,
    test_data: List[Dict[str, Any]],
    method_name: str
) -> Dict[str, float]:
    """Evaluate a search method on test data."""
    logger.info(f"Evaluating {method_name}...")

    total_queries = len(test_data)
    precision_at_1 = 0
    precision_at_3 = 0
    precision_at_5 = 0
    recall_at_5 = 0
    latencies = []

    for item in test_data:
        query = item['query']
        relevant_ids = set(str(rid) for rid in item['relevant_doc_ids'])

        if not relevant_ids:
            continue  # Skip queries without ground truth

        # Time the search
        start_time = time.time()
        try:
            results = search_function(query)
            latency = time.time() - start_time
            latencies.append(latency)

            # Extract document IDs from results
            retrieved_ids = set()
            for doc in results:
                # Try to extract ID from metadata or content hash
                doc_id = doc.metadata.get('id')
                if doc_id is None:
                    # Fallback: use content hash as ID
                    import hashlib
                    doc_id = hashlib.md5(doc.page_content.encode()).hexdigest()[:8]
                retrieved_ids.add(str(doc_id))

            # Calculate metrics
            retrieved_list = list(retrieved_ids)

            # Precision@1
            if len(retrieved_list) >= 1 and retrieved_list[0] in relevant_ids:
                precision_at_1 += 1

            # Precision@3
            relevant_retrieved_3 = sum(1 for rid in retrieved_list[:3] if rid in relevant_ids)
            precision_at_3 += relevant_retrieved_3 / min(3, len(retrieved_list)) if retrieved_list else 0

            # Precision@5
            relevant_retrieved_5 = sum(1 for rid in retrieved_list[:5] if rid in relevant_ids)
            precision_at_5 += relevant_retrieved_5 / min(5, len(retrieved_list)) if retrieved_list else 0

            # Recall@5
            recall_at_5 += len(relevant_ids.intersection(retrieved_ids)) / len(relevant_ids) if relevant_ids else 0

        except Exception as e:
            logger.error(f"Error evaluating query '{query}': {e}")
            latencies.append(float('inf'))

    # Calculate averages
    metrics = {
        'precision@1': precision_at_1 / total_queries if total_queries > 0 else 0,
        'precision@3': precision_at_3 / total_queries if total_queries > 0 else 0,
        'precision@5': precision_at_5 / total_queries if total_queries > 0 else 0,
        'recall@5': recall_at_5 / total_queries if total_queries > 0 else 0,
        'avg_latency_ms': sum(latencies) / len(latencies) * 1000 if latencies else float('inf')
    }

    logger.info(f"{method_name} Results:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")

    return metrics


def dense_search_function(query: str) -> List[Document]:
    """Dense search for comparison."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = QdrantClient("http://localhost:6333")

    qry_vec = model.encode(query).tolist()
    results = client.query_points(
        collection_name="simple_rag",
        query=qry_vec,
        limit=5,
        with_payload=True
    )

    documents = []
    for hit in results.points:
        if isinstance(hit.payload, dict) and 'text' in hit.payload:
            documents.append(Document(
                page_content=hit.payload['text'],
                metadata={'id': hit.id, 'score': hit.score}
            ))
    return documents


def hybrid_search_function(query: str) -> List[Document]:
    """Hybrid search function."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = QdrantClient("http://localhost:6333")

    sparse_encoder = BM25Encoder()
    try:
        sparse_encoder.load_vocab('data/sparse_vocab.json')
    except FileNotFoundError:
        logger.warning("Sparse vocab not found, using dense fallback")
        return dense_search_function(query)

    retriever = HybridRetriever(
        qdrant_client=client,
        dense_model=model,
        sparse_encoder=sparse_encoder,
        collection_name="simple_rag_hybrid",  # Use hybrid collection
        top_k=5
    )

    return retriever.search(query)


def hybrid_rerank_search_function(query: str) -> List[Document]:
    """Hybrid search with reranking."""
    docs = hybrid_search_function(query)

    if not docs:
        return docs

    try:
        reranker = create_reranker()
        return reranker.rerank(query, docs, top_k=5)
    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        return docs


def main():
    """Main evaluation function."""
    logging.basicConfig(level=logging.INFO)

    # Load test data
    test_data = load_test_data()

    # Define search methods to evaluate
    search_methods = [
        ("Dense Search", dense_search_function),
        ("Hybrid Search", hybrid_search_function),
        ("Hybrid + Rerank", hybrid_rerank_search_function),
    ]

    results = {}

    # Evaluate each method
    for method_name, search_func in search_methods:
        try:
            metrics = evaluate_search_method(search_func, test_data, method_name)
            results[method_name] = metrics
        except Exception as e:
            logger.error(f"Failed to evaluate {method_name}: {e}")
            results[method_name] = {'error': str(e)}

    # Print comparison
    print("\n" + "="*60)
    print("EVALUATION RESULTS COMPARISON")
    print("="*60)

    print("<20")
    for method_name, metrics in results.items():
        if 'error' in metrics:
            print(f"{method_name:<20} ERROR: {metrics['error']}")
        else:
            print("<20")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()