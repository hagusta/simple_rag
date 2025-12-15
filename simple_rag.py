import chainlit as cl
from typing import List
from langchain_core.messages import AIMessage, HumanMessage
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.documents import Document

from langchain_openai import ChatOpenAI

from operator import itemgetter

from pydantic import SecretStr

model = SentenceTransformer("all-MiniLM-L6-v2")

LLM_SERVER='http://0.0.0.0:8081/v1'
LLM_API_KEY='sk-no-key-required'
LLM_MODEL='ibm-granite/granite-4.0-micro'

VECTOR_SERVER="http://localhost:6333"
VECTOR_COLLECTION='simple_rag'
VECTOR_TOP_K=5
VECTOR_SCORE_THRESHOLD=0.7

llm = ChatOpenAI(base_url=LLM_SERVER,
                 api_key=SecretStr(LLM_API_KEY),
                 stream_usage=True,
                )

system_content='''
    You are an usefull assistance who answer user query.

    Prioritize answer by checking in the contexts given.

    Do not make up Answer. Answer "I have no knoweledge of your query" if you lack of knoweledge. Anwser "I do not have the information" if you lack of information   
    '''

chat_history:List=[]

def get_qdrant_client():
    """Create a singleton Qdrant client."""
    return QdrantClient(VECTOR_SERVER)

def context_retreiver(
    query: str
    ) -> List[Document]:
    """Perform semantic search on code chunks."""
    qry_vec = model.encode(query).tolist() 
    client = get_qdrant_client()
    top_k=VECTOR_TOP_K

    try:
        results = client.query_points(
            collection_name=VECTOR_COLLECTION, 
            query=qry_vec, 
            limit=top_k,
            #score_threshold=VECTOR_SCORE_THRESHOLD,
        )
        #print(results)
        return [ Document(hit.payload['text']) for hit in results.points]
    except Exception as e:
        print(f"Search error: {e}")
        return []

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

contextualize_question_prompt = (
    ChatPromptTemplate.from_messages([
        ("system",contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user","{query}")
        ])
        )

def contextualize_question(x):
    #if '<debug>' in x.lower():
    print("contextualize_question:",x)
    return x.content

history_aware_retriever = (
        contextualize_question_prompt
        | llm
        | RunnableLambda(contextualize_question) 
        | RunnableLambda(context_retreiver) )

system_prompt='''
You are an usefull assistance who answer user query.

Prioritize answer by checking in the contexts given.

Do not make up Answer. Answer "I have no knoweledge of your query" if you lack of knoweledge. Anwser "I do not have the information" if you lack of information   
'''

user_prompt = """
### contexts
{context}

create answer from ### contexts to user query:
### question
{query}
"""

prompt=(
    RunnableParallel({
        "query": itemgetter("query"),
        "context": history_aware_retriever,
        })
    | ChatPromptTemplate.from_messages([
        ("system",system_prompt),
        ("user",user_prompt)
       ])
    )


chain = (
    RunnableParallel({
        "query":itemgetter("query"),
        "chat_history":itemgetter("chat_history")
        }
        )
    | prompt
    | llm
        )

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("chat_history",[])

@cl.on_message
async def on_message(message: cl.Message):
    chat_history=cl.user_session.get("chat_history")
    
    if chat_history is None:
        chat_history=[]

    result = await history_aware_retriever.ainvoke({"query":message.content,"chat_history":chat_history})

    #print(f"result:\n{result}")

    stream = chain.astream({"query":message.content,"chat_history":chat_history})

    msg = await cl.Message(content="").send()

    assistance_respond = ""

    async for chunk in stream:
        if token := chunk.text:
            assistance_respond=assistance_respond+token
            await msg.stream_token(token)
    
    chat_history.append(HumanMessage(message.content))
    chat_history.append(AIMessage(assistance_respond))
    #print(chat_history)

    await msg.update()

