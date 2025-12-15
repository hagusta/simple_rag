from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
from typing import List
import random
import json
from json.decoder import JSONDecodeError
import pandas as pd

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


def main():
    """Create question and answer from text
       first create text chunks from document using Langchain recursive text split.
       then passes text chunks to KIMI to produce relevant questions and corresponding answer from the text chunk 
       store text chunk, question and answer
    """

    #LLM_SERVER='http://0.0.0.0:8081/v1'
    #LLM_API_KEY='sk-no-key-required'
 
#LLM_MODEL='TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    LLM_SERVER='https://openrouter.ai/api/v1'
    LLM_API_KEY=os.environ["OPENROUTER_API_KEY"]
    LLM_MODEL='moonshotai/kimi-k2:free'

    #print(LLM_API_KEY)

    with open("data/the_republic_introduction.txt") as f:
        doc = f.read()

    chunks = document_to_chunks([doc])

    system_role='''
You are to create questions and answers (Q&A) from a given text.

### INSTRUCTION:
1. Identifies proper noun in the text.
2. Derive Q&A from the identified proper nouns with relevant information from the text 

### EXAMPLE:

text :
The Republic of Plato is the longest of his works with the exception of the Laws, and is certainly the greatest of them. 
There are nearer approaches to modern metaphysics in the Philebus and in the Sophist; the Politicus or Statesman is more ideal; the form and institutions of the State are more clearly drawn out in the Laws; as works of art, the Symposium and the Protagoras are of higher excellence. 
But no other Dialogue of Plato has the same largeness of view and the same perfection of style; no other shows an equal knowledge of the world, or contains more of those thoughts which are new as well as old, and not of one age only but of all. 
Nowhere in Plato is there a deeper irony or a greater wealth of humour or imagery, or more dramatic power. 
Nor in any other of his writings is the attempt made to interweave life and speculation, or to connect politics with philosophy. 

Simple Q&A example from text:

Q: What is the Republic?
A: One of Plato writing.

Q: What is the Republic?
A: One of Dialogue of Plato.

Q: What is the Republic?
A: Plato greatest works.

Q: What are other Plato works?
A: The Philebus is other works.

Complex Q&A example from text:

Q: How is The Republic significant?
A: No other Plato works have: the largestness of view, perfection of style, show equal knowledge of the world, deeper irony, humour, imagery and dramatic power.

Q: What are other Plato works?
A: Other Plato works includes: the Laws, the Philebus, the Sophist, the Politicus, Statesman, the Symposium and the Protagoras,

### OUTPUT:
Expected output is in json without json markdown

{
    "text" : "The Republic of Plato is the longest of his works with the exception of the Laws, and is certainly the greatest of them. 
There are nearer approaches to modern metaphysics in the Philebus and in the Sophist; the Politicus or Statesman is more ideal; the form and institutions of the State are more clearly drawn out in the Laws; as works of art, the Symposium and the Protagoras are of higher excellence. 
But no other Dialogue of Plato has the same largeness of view and the same perfection of style; no other shows an equal knowledge of the world, or contains more of those thoughts which are new as well as old, and not of one age only but of all. 
Nowhere in Plato is there a deeper irony or a greater wealth of humour or imagery, or more dramatic power. 
Nor in any other of his writings is the attempt made to interweave life and speculation, or to connect politics with philosophy."
    "results" : [
        {
            "q" : "What is the Republic?",
            "a" : "One of Plato writing."
        },
        {
            "q" : "What is the Republic?",
            "a" : "One of Dialogue of Plato."
        },
        {
            "q" : "What is the Republic",
            "a" : "Plato greatest works."
        },
        {
            "q" : "What are other Plato works?",
            "a" : "The Philebus is other works."
        },
        {
            "q" : "How is The Republic significant?",
            "a" : "No other Plato works have: the largestness of view, perfection of style, show equal knowledge of the world, deeper irony, humour, imagery and dramatic power."
        },
         {
            "q" : "What are other Plato works?",
            "a" : "Other Plato works includes: the Laws, the Philebus, the Sophist, the Politicus, Statesman, the Symposium and the Protagoras."
        },  
    ]
}

### VERIFICATION:
Verify output:
1. make sure format in json
2. make sure the Q&A subject is proper noun not common noun (for example: Plato not the philosoper, Greek not the country, The Republic not the book )
3. check the Q&A relevant to the text
4. make sure to include the original text(chat)
5. make sure to not include ```json markdown
6. do not create repeat Q&A
   '''

    results = []

    chunck_count=len(chunks)

    ridx = [ random.randint(0,chunck_count) for _ in range(int(chunck_count/20)) ]
    client = OpenAI(base_url=LLM_SERVER, api_key=LLM_API_KEY)

    i=0

    for i,idx in enumerate(ridx):
        print(i,idx)
        messages = [
                {"role": "system", "content": system_role},
                {"role": "user", "content": chunks[idx] }
            ]
        responses = client.chat.completions.create(model=LLM_MODEL,messages=messages)
        content = responses.choices[0].message.content.replace('```json','').replace('```','')
        #print(content)
        try:
            data = json.loads(content)
            data['text_id']=idx
            results.append(data)
        except (JSONDecodeError,TypeError):
            print(JSONDecodeError,TypeError)
            print(content)
            continue
    
    normalized = list()
    df=pd.DataFrame.from_records(results)
    df = df.apply(lambda x: x.explode()).reset_index(drop=True)
    d = pd.json_normalize(df['results'], sep='_')
    d.columns = ["question","answer"]
    normalized.append(d.copy())
    df = pd.concat([df] + normalized, axis=1).drop('results', axis=1)
    df.index.name='idx'

    df.to_csv('data/qa.csv')

if __name__ == "__main__":
    main()







