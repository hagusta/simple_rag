from rich.console import Console
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from typing import Dict,List

model = SentenceTransformer("all-MiniLM-L6-v2")
collection_name='rag_test'

def get_qdrant_client():
    """Create a singleton Qdrant client."""
    return QdrantClient("http://localhost:6333")

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
                "context": hit.payload["text"],
            }
            for hit in results
        ]
    except Exception as e:
        print(f"Search error: {e}")
        return []

def main():
    LLM_SERVER='http://0.0.0.0:8081/v1'
    LLM_API_KEY='sk-no-key-required'
    LLM_MODEL='TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    #LLM_SERVER='https://openrouter.ai/api/v1'
    #LLM_API_KEY=os.environ["OPENROUTER_API_KEY"]
    #LLM_MODEL='moonshotai/kimi-k2:free'
   
    client = OpenAI(base_url=LLM_SERVER, api_key=LLM_API_KEY)

    console = Console()

    system_content='''
    You are an usefull assistance who answer user query.

    Prioritize answer by checking in the contexts given.

    Do not make up Answer. Answer "I have no knoweledge of your query" if you lack of knoweledge. Anwser "I do not have the information" if you lack of information   
    '''
    
    chat_history=[]

    while True: 
        query=console.input("What is [i]your[/i] [bold red]query[/]? :smiley: ")
        if query.lower() in ["quit", "exit", "bye"]:
            break

        if query.lower() == 'clear':
            chat_history = []
            continue

        contexts=semantic_search(query,collection_name,10)

        user_content=f'''
        ###context:
        {contexts}

        create answer from ###context to user query:
        {query}

        and cite the section of the ###context
        '''

        if len(chat_history) > 0:
            chat="\n".join(chat_history)
            user_content=user_content + \
            f'''
            ###chat_history:
            {chat}
            '''
        #print(user_content)

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query}
        ]

        responses = client.chat.completions.create(model=LLM_MODEL,messages=messages)

        console.print(responses.choices[0].message.content)

        chat_history.append(query)
        chat_history.append(responses.choices[0].message.content)

if __name__ == "__main__":
    main()
