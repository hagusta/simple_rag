from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
from typing import List
import random
import json
from json.decoder import JSONDecodeError
import pandas as pd
import pickle

def main():
    """Create question and answer from text
       first create text chunks from document using Langchain recursive text split.
       then passes text chunks to KIMI to produce relevant questions and corresponding answer from the text chunk 
       store text chunk, question and answer
    """

    LLM_SERVER='https://openrouter.ai/api/v1'
    LLM_API_KEY=os.environ["OPENROUTER_API_KEY"]
    LLM_MODEL='moonshotai/kimi-k2:free'

    print("load chunks . . .")

    with open("data/chunks_90.pkl","rb") as f:
        chunks = pickle.load(f)


    with open('prompt/system_prompt.ds.txt') as f:
        system_role = f.read()

    results = []

    chunk_count=len(chunks)

    print(f"{chunk_count} chunks loaded . . .")

    ridx = [ random.randint(0,chunk_count) for _ in range(int(chunk_count/20)) ]
    client = OpenAI(base_url=LLM_SERVER, api_key=LLM_API_KEY)

    i=0

    print("start create question and answer questions. . .")

    for i,idx in enumerate(ridx):
        print(i,idx,len(chunks[idx][2]))
        if len(chunks[idx][2]) < 20:
            continue
        messages = [
                {"role": "system", "content": system_role},
                {"role": "user", "content": chunks[idx][2] }
            ]
        responses = client.chat.completions.create(model=LLM_MODEL,messages=messages)
        content = responses.choices[0].message.content.replace('```json','').replace('```','')
        try:
            data = json.loads(content)
            data['text_id']=idx
            data['start']=chunks[idx][0]
            data['end']=chunks[idx][1]
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

    print(f"{df.count()} question and answer pair are created")

    df.to_csv('data/qa.csv')

if __name__ == "__main__":
    main()







