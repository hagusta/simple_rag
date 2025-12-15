from sentence_transformers import SentenceTransformer

import numpy as np
from typing import List

import pickle

filename='/home/hagusta/workspace/python_sandbox/data/the_republic_introduction.txt'

def split_sentences(text:str) -> List:
    sentence_stop=". "
    paragraph_delimiter = "\n\n"
    delimiter_on=True
    sentences=[]
    sentence=""
    last_stop=0
    for i,ch in enumerate(text):
        if (sentence_stop in sentence and delimiter_on) or (paragraph_delimiter in sentence):
            sentences.append([last_stop,i-1,sentence.replace(". ",".")])
            last_stop=i-1
            delimiter_on=True
            sentence=ch
            continue
        if "(" in sentence:
            delimiter_on=False
        sentence=sentence+ch
    return sentences

norm=np.linalg.norm

def cosine_distance(m:np.ndarray,v:np.ndarray)->np.ndarray:
    return m.dot(v)/norm(m)/norm(v)

def main():
    with open(filename,'r') as f:
        text=f.read()

    chunks=split_sentences(text)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    sentences_distance=[0]
    embed=[]
    for chunk in chunks:
        embed.append(model.encode(chunk[2]))
        if len(embed) > 1:
            sentences_distance.append(cosine_distance(embed[len(embed)-2],embed[len(embed)-1]))

    ave=np.mean(sentences_distance)
    sigma=np.std(sentences_distance)
    ninety_percent=[ [i,dist] for i,dist in enumerate(sentences_distance) if dist > (ave + 1.28 * sigma) or dist < (ave - 1.28 * sigma) ]

    chunks_90=[]
    lidx=0
    for idx,_ in ninety_percent:
        sent=""
        if idx == 0:
            continue
        for chunk in chunks[lidx:idx]:
            sent= sent + chunk[2]
        #print(lidx,idx,sent,"\n")
        chunks_90.append([chunks[lidx][0],chunks[idx-1][1],sent])
        lidx=idx

    with open('data/chunks_90.pkl','wb') as f:
        pickle.dump(chunks_90,f)

if __name__ == "__main__":
    main()
