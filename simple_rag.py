import atexit
import json
import logging.config
import pathlib
import logging
import os

import chainlit as cl
from typing import List,Dict
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompt_values import ChatPromptValue
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnableBranch
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.documents import Document

from langchain_openai import ChatOpenAI

from operator import itemgetter

from pydantic import SecretStr

# Import hybrid search components
from hybrid_search import HybridRetriever
from reranker import create_reranker
from sparse_utils import BM25Encoder

logger = logging.getLogger(__name__)  # __name__ is a common choice

#LLM_SERVER='http://0.0.0.0:8081/v1'
#LLM_API_KEY='sk-no-key-required'
#LLM_MODEL='ibm-granite/granite-4.0-micro'

LLM_SERVER = os.getenv('LLM_SERVER', 'https://openrouter.ai/api/v1')
LLM_API_KEY = os.getenv('LLM_API_KEY', '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')
LLM_MODEL = os.getenv('LLM_MODEL', 'x-ai/grok-4.1-fast:free')

VECTOR_SERVER = os.getenv('VECTOR_SERVER', 'http://localhost:6333')
VECTOR_COLLECTION = os.getenv('VECTOR_COLLECTION', 'simple_rag_hybrid')
VECTOR_TOP_K = int(os.getenv('VECTOR_TOP_K', '20'))
VECTOR_SCORE_THRESHOLD = float(os.getenv('VECTOR_SCORE_THRESHOLD', '0.7'))

# Hybrid search configuration
HYBRID_SEARCH_ENABLED = os.getenv('HYBRID_SEARCH_ENABLED', 'true').lower() == 'true'
RERANKER_ENABLED = os.getenv('RERANKER_ENABLED', 'true').lower() == 'true'
RERANK_TOP_K = int(os.getenv('RERANK_TOP_K', '20'))
HYBRID_DENSE_WEIGHT = float(os.getenv('HYBRID_DENSE_WEIGHT', '0.7'))
HYBRID_SPARSE_WEIGHT = float(os.getenv('HYBRID_SPARSE_WEIGHT', '0.3'))

DEBUG:bool=False

def get_handler_by_name(handler_name):
    """Manual workaround for older Python versions."""
    for handler in logging.root.handlers:
        if hasattr(handler, 'name') and handler.name == handler_name:
            return handler
    # Check other loggers if necessary
    return None

def setup_logging():
    config_file = pathlib.Path("config/logging_configs.json")
    with open(config_file) as f_in:
        config = json.load(f_in)
    #print(config)
    config['handlers']['file']['filename']="logs/simple_rag.jsonl"
    logging.config.dictConfig(config)
    queue_handler = get_handler_by_name("queue_handler")
    if queue_handler is not None:
        queue_handler.listener.start()
        atexit.register(queue_handler.listener.stop)

model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize hybrid search components
hybrid_retriever = None
reranker = None

if HYBRID_SEARCH_ENABLED:
    try:
        # Initialize sparse encoder
        sparse_encoder = BM25Encoder()
        # Try to load existing vocabulary, fallback to fitting on empty corpus
        try:
            sparse_encoder.load_vocab('data/sparse_vocab.json')
            logger.info("Loaded existing sparse vocabulary")
        except FileNotFoundError:
            logger.warning("Sparse vocabulary not found, sparse search will be limited")
            sparse_encoder.fit([])  # Initialize with empty corpus

        # Initialize hybrid retriever
        hybrid_retriever = HybridRetriever(
            qdrant_client=QdrantClient(VECTOR_SERVER),
            dense_model=model,
            sparse_encoder=sparse_encoder,
            collection_name=VECTOR_COLLECTION,
            dense_weight=HYBRID_DENSE_WEIGHT,
            sparse_weight=HYBRID_SPARSE_WEIGHT,
            top_k=VECTOR_TOP_K
        )
        logger.info("Hybrid search initialized")
    except Exception as e:
        logger.error(f"Failed to initialize hybrid search: {e}")
        logger.info("Falling back to dense search only")

if RERANKER_ENABLED:
    try:
        reranker = create_reranker()
        logger.info("Reranker initialized")
    except Exception as e:
        logger.error(f"Failed to initialize reranker: {e}")
        logger.info("Reranker disabled")

llm = ChatOpenAI(base_url=LLM_SERVER,
                 api_key=SecretStr(LLM_API_KEY),
                 stream_usage=True,
                )

chat_history:List=[]

def get_qdrant_client():
    """Create a singleton Qdrant client."""
    return QdrantClient(VECTOR_SERVER)

def context_retreiver(
    query: str
    ) -> List[Document]:
    """Perform hybrid search with optional reranking on code chunks."""
    try:
        documents = []

        # Try hybrid search first
        if hybrid_retriever is not None:
            documents = hybrid_retriever.search(query)
            logger.debug(f"Hybrid search returned {len(documents)} results")
        else:
            # Fallback to dense search
            logger.info("Using dense search (hybrid search not available)")
            documents = dense_search_fallback(query)

        # Apply reranking if enabled
        if reranker is not None and documents:
            # Get more candidates for reranking
            if hybrid_retriever is not None:
                extended_results = hybrid_retriever.search(query)  # Get more results for reranking
                if len(extended_results) > len(documents):
                    documents = extended_results[:RERANK_TOP_K]  # Take top candidates for reranking

            documents = reranker.rerank(query, documents, top_k=VECTOR_TOP_K)
            logger.debug(f"Reranking applied, final results: {len(documents)}")

        return documents

    except Exception as e:
        logger.error(f"Search error: {e}")
        # Final fallback to basic dense search
        return dense_search_fallback(query)


def dense_search_fallback(query: str) -> List[Document]:
    """Fallback dense search function."""
    qry_vec = model.encode(query).tolist()
    client = get_qdrant_client()
    top_k = VECTOR_TOP_K

    try:
        results = client.query_points(
            collection_name=VECTOR_COLLECTION,
            query=qry_vec,
            limit=top_k,
        )
        return [
            Document(
                page_content=hit.payload['text'],
                metadata={'score': hit.score, 'search_type': 'dense_fallback'}
            )
            for hit in results.points
        ]
    except Exception as e:
        logger.error(f"Dense search fallback error: {e}")
        return []

contextualize_q_system_prompt = """Given a chat history and the latest user question 
which might reference context in the chat history, formulate a standalone question 
which can be understood without the chat history. 
Do NOT answer the question, just reformulate it if needed and otherwise return it as is."""

contextualize_question_prompt = (
    ChatPromptTemplate.from_messages([
        ("system",contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user","{query}")
        ])
        )

def inspect_prompt(x:ChatPromptValue) -> ChatPromptValue:
    DEBUG = cl.user_session.get("DEBUG")
    if DEBUG:
        #print(f"prompt:{'-' * 50 }\n {x}\n")
        #print('-' * 60,'\n')
        logger.debug(f"{inspect_prompt.__name__}",extra={"prompt":f"{x}"})
    return x

def inspect_contextualize_question(x:AIMessage)->str:
    question=""
    if isinstance(x,list):
        question="\n".join(x)
    if isinstance(x.content,str):
        question=x.content
    DEBUG=cl.user_session.get("DEBUG")
    if DEBUG:
        #print(f"contextualize_question:{'-'*50}\n{question}\n")
        #print('-' * 60,'\n')
        logger.debug(f"{inspect_contextualize_question.__name__}",extra={"question":f"{question}"})
    return question

def inspect_context_retreived(x:list[Document])->list[Document]:
    DEBUG=cl.user_session.get("DEBUG")
    if DEBUG:
        #print("context_retreived:",'-'*50)
        #_=[ print(f"doc:\n{doc.page_content}.") for doc in x ]
        #print('-' * 60,'\n')
        retrieved = [f"doc{i}:\n{doc.page_content}" for i,doc in enumerate(x)]
        retrieved = "\n".join(retrieved)
        logger.debug(f"{inspect_context_retreived.__name__}",extra={
            "content": retrieved
            })
    return x

def is_historical(x:Dict)->bool:
    DEBUG=cl.user_session.get("DEBUG")
    is_hist=len(x['chat_history']) > 0
    branch='non_historic_retreival'
    if is_hist:
        branch='historic_retreival'
    if DEBUG:
        logger.debug(f"{is_historical.__name__}",extra={"branch":branch})
    if is_hist:
        return True
    return False

non_historic_retreiver = (
        itemgetter("query")
        | RunnableLambda(context_retreiver)
        | RunnableLambda(inspect_context_retreived)
        )

historic_retriever = (
        contextualize_question_prompt
        | RunnableLambda(inspect_prompt)
        | llm
        | RunnableLambda(inspect_contextualize_question) 
        | RunnableLambda(context_retreiver) 
        | RunnableLambda(inspect_context_retreived)
        )

history_aware_retriever = (
        RunnableBranch((RunnableLambda(is_historical),historic_retriever),non_historic_retreiver)
        )


system_prompt='''
You are an usefull assistance who answer user query.

Answer only from ###contexts given.

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
    | RunnableLambda(inspect_prompt)
    | llm
        )

setup_logging() 

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("chat_history",[])
    cl.user_session.set("DEBUG",False)

@cl.on_message
async def on_message(message: cl.Message):
    chat_history=cl.user_session.get("chat_history")
    if '<debug>' in message.content.lower():
        query=message.content.lower().replace('<debug>','')
        DEBUG=True
        cl.user_session.set("DEBUG",True)
    else:
        query=message.content
        DEBUG=False
        cl.user_session.set("DEBUG",False)

    
    if chat_history is None:
        chat_history=[]

    #_ = await history_aware_retriever.ainvoke({"query":query,"chat_history":chat_history})

    #print(f"result:\n{result}")

    stream = chain.astream({"query":query,"chat_history":chat_history})

    msg = await cl.Message(content="").send()

    assistance_respond = ""

    async for chunk in stream:
        if token := chunk.text:
            assistance_respond=assistance_respond+token
            await msg.stream_token(token)
    
    chat_history.append(HumanMessage(query))
    chat_history.append(AIMessage(assistance_respond))
    #print(chat_history)

    if DEBUG:
        hist=[]
        for block in chat_history:
            if isinstance(block,AIMessage):
                hist.append(f"AIMessage: {block.content}")
            else:
                hist.append(f"Human: {block.content}")
        logger.debug("on_message",extra={"messages_history":hist})

    await msg.update()

