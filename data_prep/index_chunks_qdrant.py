from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from typing import List,Dict
import uuid

collection_name='rag_test'

with open("data/the_republic_introduction.txt") as f:
    doc = f.read()

model = SentenceTransformer("all-MiniLM-L6-v2")

def get_qdrant_client():
    """Create a singleton Qdrant client."""
    return QdrantClient("http://localhost:6333")

client=get_qdrant_client()

def document_to_chunks (documents:List):
    split_to_chunk = RecursiveCharacterTextSplitter(
            chunk_size = 300,
            chunk_overlap = 0,
            length_function = len,
            strip_whitespace = True,
            separators=["\n\n"]
            )
    chunks=[]
    for doc in documents:
        for chunk in split_to_chunk.split_text(doc):
            chunks.append(chunk.replace("\n"," ").replace("  "," "))
    return chunks

def main():
    """Break given document to chunck 
       get embedded vector for each chunck
       upload/upsert to qdrant
       test query to qdrant
    """

    i = 0
    
    #create qdrant collection
    try:
        client.get_collection(collection_name)
        client.delete_collection(collection_name)
        print(f"create qdrant collection ...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=model.get_sentence_embedding_dimension(), distance=Distance.COSINE),
        )
        print(f"qdrant collection {collection_name} is created")
    except Exception as e:
        print(f"{e}\ncreate qdrant collection ...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=model.get_sentence_embedding_dimension(), distance=Distance.COSINE),
        )
        print(f"qdrant collection {collection_name} is created")
    
    #create point data from chuncks
    print("start document chunking and qdrant data point creation . . . ")
    Points =[
            PointStruct(
                id = str(uuid.uuid1()),
                vector = model.encode(chunk),
                payload = {
                    "text" : chunk,
                    "book" : "the republic",
                    "file_path" : "data/the_republic_introduction.txt"
                    },
                )
           for i,chunk in enumerate(document_to_chunks([doc]))
    ]

    #upsert to qdrant
    client.upsert(collection_name=collection_name, points=Points)
    print(f"{len(Points)} data point created and inserted to qdrant {collection_name}")

    #inspect chunks no 5
    print("inspect chuck no 5 for test")
    for chunk in document_to_chunks([doc]):
        if i == 5:
            print(i,len(chunk))
            embedding = model.encode([chunk])
            print(embedding.shape)
        i=i+1

    #test semantic_search
    query="what is the Republic?"

    print(f"test qdrant query: \n{query}\n")

    res = semantic_search(query=query,collection_name=collection_name)

    print(res)

def semantic_search(
    query: str, collection_name: str, top_k: int = 5
) -> List[Dict]:
    """Perform semantic search on code chunks."""
    contexts=[]
    qry_vec = model.encode(query).tolist() 
    client = get_qdrant_client()

    try:
        results = client.search(
            collection_name=collection_name, query_vector=qry_vec, limit=top_k
        )

        return [
            {
                "file_path": hit.payload["file_path"],
                "book": hit.payload["book"],
                "text": hit.payload["text"],
                "score": hit.score,
            }
            for hit in results
        ]
    except Exception as e:
        print(f"Search error: {e}")
        return []

if __name__ == "__main__" :
    main()
