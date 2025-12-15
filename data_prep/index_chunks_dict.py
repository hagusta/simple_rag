from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle 
import heapq as q

from typing import List,Dict

collection_name='rag_test'

with open("data/the_republic_introduction.txt") as f:
    doc = f.read()

vector=[]
vector_payload=[]

model = SentenceTransformer("all-MiniLM-L6-v2")

@dataclass
class Point:
    idx:int
    pid:int
    vector:List 
    payload:Dict

def vectors_upsert(points:List[Point]):
    for point in points:
        vector.append(point.vector)
        vector_payload.append(point)
    vector_np=np.array(vector)
    with open('data/vector_space.pkl','wb') as f:
        pickle.dump(vector_np,f)
    with open('data/vertor_load.pkl','wb') as f:
        pickle.dump(vector_payload,f)

    return True

norm = np.linalg.norm

def vectors_search(query,k:int)->List:
    vector_query=np.array(model.encode(query).tolist())
    print(f"vector_query shape {vector_query.shape}")
    with open('data/vector_space.pkl','rb') as f:
        vector_space = np.array(pickle.load(f))
    print(f"vector_spaece shape {vector_space.shape}")
    y=vector_space.dot(vector_query)/(norm(vector_space) * norm(vector_query))
    heap=[]
    _ = [q.heappush(heap,(score,{'idx':i,'score':score,'emb':vector_space[i]})) for i,score in enumerate(y) ]    
    return q.nlargest(k,heap) 

def document_to_chunks (documents:List):
    split_to_chunk = RecursiveCharacterTextSplitter(
            chunk_size = 400,
            chunk_overlap = 0,
            length_function = len,
            strip_whitespace = True,
            separators=["\n\n"]
            )
    chunks=[]
    for doc in documents:
        for chunk in split_to_chunk.split_text(doc):
            chunks.append(chunk)
    return chunks

def chunks_to_token(chunk:str):
    return model.encode(chunk)

def main():
    """Break given document to chunck 
       get embedded vector for each chunck
       upload/upsert to simple vector search 
       test query to vector space 
    """
    
    #create point data from chuncks
    print("start document chunking and data point creation . . . ")
    Points =[
            Point(
                idx=i,
                pid = hash(frozenset(model.encode(chunk).tolist())),
                vector = model.encode(chunk.replace('\n',' ').replace('  ',' ')).tolist(),
                payload = {
                    "text" : chunk.replace('\n',' ').replace('  ',' '),
                    "book" : "the republic",
                    "file_path" : "data/the_republic_introduction.txt"
                    },
                )
           for i,chunk in enumerate(document_to_chunks([doc]))
    ]

    print("create vector space . . .") 
    vectors_upsert(points=Points)
    print(f"{len(Points)} data point created and inserted to vector search engine\n")

    #test semantic_search
    query="who is bendis?"

    print(f"test qdrant query: \n{query}\n")

    res = vectors_search(query=query,k=10)

    #print(res)

    for r in res:
        print(f"index: {r[1]['idx']} score: {r[0]}")
        print(f"{vector_payload[r[1]['idx']].payload}\n")

if __name__ == "__main__" :
    main()
