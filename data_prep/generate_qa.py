from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from typing import List
import random
import json
from json.decoder import JSONDecodeError
import pandas as pd
import time

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
            chunks.append(chunk)
    return chunks


def main():
    """Create question and answer from text
       first create text chunks from document using Langchain recursive text split.
       then passes text chunks to KIMI to produce relevant questions and corresponding answer from the text chunk 
       store text chunk, question and answer
    """

    LLM_SERVER='http://0.0.0.0:8081/v1'
    LLM_API_KEY='sk-no-key-required'
    LLM_MODEL='ibm-granite/granite-4.0-micro'

    #LLM_SERVER='https://openrouter.ai/api/v1'
    #LLM_API_KEY=os.environ["OPENROUTER_API_KEY"]
    #LLM_MODEL='moonshotai/kimi-k2:free'

    with open("data/the_republic_introduction.txt") as f:
        doc = f.read()

    chunks = document_to_chunks([doc])

    with open('prompt/system_prompt.ds.txt') as f:
        system_role = f.read()

    results = []

    chunck_count=len(chunks)-1

    exclude=[310, 349, 378, 297, 152, 313, 180, 5, 135, 46, 357, 183, 294, 133, 210, 35, 351, 331, 352, 123, 84, 186, 237, 364, 238, 358, 268, 100, 31, 174, 315, 134, 274, 153, 69, 57, 291, 48, 52, 93, 159, 9, 298, 11]

    ridx = [ random.randint(0,chunck_count) for _ in range(int(chunck_count/20)) ]
    ridx = [ id for id in ridx if id not in exclude ]
    client = OpenAI(base_url=LLM_SERVER, api_key=LLM_API_KEY)

    i=0

    #ridx = [327] + ridx

    print(f"chunks count {chunck_count}")

    for i,idx in enumerate(ridx):
        print(i,idx,len(chunks[idx]))
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

    df.to_csv(f"data/question_answer.{time.time()}.csv")

if __name__ == "__main__":
    main()
