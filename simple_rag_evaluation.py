from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

import pickle

from typing import Dict,List

def get_qdrant_client(model:SentenceTransformer,collection_name:str):
    """Create a singleton Qdrant client."""
    client =QdrantClient("http://localhost:6333")
    try:
        client.delete_collection(collection_name=collection_name)
        print(f"{collection_name} deleted")
        print(f"create collection {collection_name} . . .")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=model.get_sentence_embedding_dimension(), distance=Distance.COSINE),
        )
        print(f"collection {collection_name} created")
    except Exception as e:
        print(f"{e}\ncollection {collection_name} does not exists")
        print(f"create collection {collection_name} . . .")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=model.get_sentence_embedding_dimension(), distance=Distance.COSINE),
        )
        print(f"collection {collection_name} created")
    return client

def load_qdrant_point(model:SentenceTransformer,chunks:Dict):
    print(f"start create {len(chunks.keys())} chunks and qdrant data point creation . . . ")
    Points =[
            PointStruct(
                id = key,
                vector = model.encode(chunk),
                payload = {
                    "text" : chunk,
                    "book" : "the republic",
                    "file_path" : "data/the_republic_introduction.txt"
                    },
                )
           for key,chunk in chunks.items()
    ]
    #upsert to qdrant
    print(f"{len(Points)} points created . . . ")
    return Points

def load_qdrant(client:QdrantClient,collection_name:str,Points:List, BATCH=1_000):
    total_count=len(Points)
    current_count=0
    while True:
        if current_count + BATCH >= total_count:
            client.upsert(collection_name=collection_name, points=Points[current_count:])
            current_count=total_count
            print(f"total point loaded: {current_count}")
            break
        elif current_count == 0:
            client.upsert(collection_name=collection_name, points=Points[:BATCH])
            current_count=current_count+BATCH
        else:
            client.upsert(collection_name=collection_name, points=Points[current_count:current_count+BATCH])
            current_count=current_count+BATCH
    print(f"total point loaded: {current_count}")

def main():
    collection_name='simple_rag'
    model = SentenceTransformer("all-MiniLM-L6-v2")

    with open('data/corpus_all.pkl','rb') as f:
        corpus = pickle.load(f)

    client=get_qdrant_client(model,collection_name)

    points = load_qdrant_point(model,chunks=corpus)

    load_qdrant(client,collection_name,points)

if __name__ == "__main__" :
    main()
