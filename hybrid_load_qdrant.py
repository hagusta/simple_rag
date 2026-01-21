"""
Hybrid data loading script for Qdrant with dense and sparse vectors.
"""
import pickle
import logging
from typing import Dict, List
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams,
    PointStruct, SparseVector
)

from sparse_utils import BM25Encoder

logger = logging.getLogger(__name__)


def create_hybrid_collection(
    client: QdrantClient,
    collection_name: str,
    dense_model: SentenceTransformer,
    sparse_encoder: BM25Encoder
) -> None:
    """Create a Qdrant collection with both dense and sparse vector support."""
    try:
        # Delete existing collection if it exists
        client.delete_collection(collection_name=collection_name)
        logger.info(f"Deleted existing collection: {collection_name}")
    except Exception:
        logger.info(f"Collection {collection_name} does not exist, creating new one")

    # Create collection with hybrid vectors
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(
                size=dense_model.get_sentence_embedding_dimension(),
                distance=Distance.COSINE
            )
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams()
        }
    )
    logger.info(f"Created hybrid collection: {collection_name}")


def load_hybrid_points(
    dense_model: SentenceTransformer,
    sparse_encoder: BM25Encoder,
    chunks: Dict
) -> List[PointStruct]:
    """Create hybrid points with both dense and sparse vectors."""
    logger.info(f"Creating hybrid vectors for {len(chunks)} chunks...")

    points = []
    for chunk_id, chunk_text in chunks.items():
        # Create dense vector
        dense_vector = dense_model.encode(chunk_text).tolist()

        # Create sparse vector
        sparse_vector = sparse_encoder.encode_document(chunk_text)

        # Convert to Qdrant sparse format
        if sparse_vector:
            sparse_indices = list(sparse_vector.keys())
            sparse_values = list(sparse_vector.values())
            sparse_qdrant = SparseVector(indices=sparse_indices, values=sparse_values)
        else:
            sparse_qdrant = SparseVector(indices=[], values=[])

        point = PointStruct(
            id=chunk_id,
            vector={
                "dense": dense_vector,
                "sparse": sparse_qdrant
            },
            payload={
                "text": chunk_text,
                "book": "the republic",
                "file_path": "data/the_republic_introduction.txt"
            }
        )
        points.append(point)

    logger.info(f"Created {len(points)} hybrid points")
    return points


def load_hybrid_data(
    client: QdrantClient,
    collection_name: str,
    points: List[PointStruct],
    batch_size: int = 1000
) -> None:
    """Load hybrid points into Qdrant collection."""
    total_points = len(points)
    current_count = 0

    logger.info(f"Loading {total_points} points into {collection_name}...")

    while current_count < total_points:
        batch_end = min(current_count + batch_size, total_points)
        batch = points[current_count:batch_end]

        try:
            client.upsert(
                collection_name=collection_name,
                points=batch
            )
            current_count = batch_end
            logger.info(f"Loaded {current_count}/{total_points} points")
        except Exception as e:
            logger.error(f"Error loading batch {current_count}-{batch_end}: {e}")
            raise

    logger.info(f"Successfully loaded {total_points} hybrid points")


def main():
    """Main function to load hybrid data."""
    collection_name = 'simple_rag_hybrid'

    # Initialize models
    logger.info("Loading models...")
    dense_model = SentenceTransformer("all-MiniLM-L6-v2")
    sparse_encoder = BM25Encoder()

    # Load corpus
    logger.info("Loading corpus data...")
    with open('data/corpus_all.pkl', 'rb') as f:
        corpus = pickle.load(f)

    # Fit sparse encoder on corpus
    logger.info("Fitting sparse encoder...")
    corpus_texts = list(corpus.values())
    sparse_encoder.fit(corpus_texts)

    # Create Qdrant client
    client = QdrantClient("http://localhost:6333")

    # Create hybrid collection
    create_hybrid_collection(client, collection_name, dense_model, sparse_encoder)

    # Create hybrid points
    points = load_hybrid_points(dense_model, sparse_encoder, corpus)

    # Load data
    load_hybrid_data(client, collection_name, points)

    # Save sparse encoder for later use
    sparse_encoder.save_vocab('data/sparse_vocab.json')
    logger.info("Saved sparse vocabulary to data/sparse_vocab.json")
    logger.info("Hybrid data loading complete!")

    # Test search
    test_query = "who is socrates?"
    logger.info(f"Testing search with query: {test_query}")

    # Test dense search
    dense_results = client.query_points(
        collection_name=collection_name,
        query=dense_model.encode(test_query).tolist(),
        limit=3,
        using='dense',
        with_payload=True
    )
    logger.info(f"Dense search results: {len(dense_results.points)}")

    # Test sparse search
    sparse_vector = sparse_encoder.encode(test_query)
    if sparse_vector:
        sparse_results = client.query_points(
            collection_name=collection_name,
            query=SparseVector(
                indices=list(sparse_vector.keys()),
                values=list(sparse_vector.values())
            ),
            limit=3,
            using='sparse',
            with_payload=True
        )
        logger.info(f"Sparse search results: {len(sparse_results.points)}")


if __name__ == "__main__":
    # Set up basic logging
    logging.basicConfig(level=logging.INFO)
    main()
