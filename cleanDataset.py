import os
import codecs
import re
from tqdm import tqdm
import itertools
from analisisCandidates import analisis_of_candidates
from candidatesExtraction import extract_candidates

MAX_LEN_CANDIDATE = 6

def clean_text(text: str) -> str:
    pattern1 = re.compile(r'\[\d+([,-]\d+)*\]')
    text = pattern1.sub("", text)   # borrado de todas las referencias del estilo [n]\[n,n]\[n-n]
    pattern2 = re.compile(r'\s[)]')
    text = pattern2.sub(')', text)
    pattern3 = re.compile(r'\s[,]')
    text = pattern3.sub(',', text)
    pattern4 = re.compile(r'\s[.]')
    text = pattern4.sub('.', text)
    pattern5 = re.compile(r'\s{2,}')
    text = pattern5.sub(" ", text)
    # TODO: delete Refs if they don´t retrieve relevant information
    # TODO: add relevant preprocssing in the future and try with different preprocessings
    # TODO: preprocess the repetitive text (el siguiente el siguient...)
    return text
    
def clean_dataset(data_path="data/docsutf8", labels_path="data/keys"):
    data = {}
    references = {}
    for dirname,dirnames ,filenames in os.walk(data_path):
        for fname in filenames:
            left = fname.split('.')[0]
            infile = os.path.join(dirname,fname)
            text = codecs.open(infile, "r", "utf-8").read()
            text = clean_text(text)
            data[left] = text.lower()
    for dirname, dirnames, filenames in os.walk(labels_path):
        for fname in filenames:
            left, right = fname.split('.')
            infile = os.path.join(dirname, fname)
            f = open(infile, 'rb')
            text = f.read().decode('utf8')
            text = text.strip()     # delete the whitespaces that are not nececessary
            #ls=text.splitlines()   # TODO: investigate references preprocessing
            text_lower = text.lower()
            references[left] = text_lower.split('\n')   
            f.close()
     
    candidates = {}
    data = dict(itertools.islice(data.items(), 5)) 
    references = dict(itertools.islice(references.items(), 5)) 
    for idx, (key, doc) in tqdm(enumerate(data.items()),desc="Calculating the candidates of each document",total=len(data)):
        candidates_of_doc = extract_candidates(doc,key)    # FORMAT CANDIDATES [(candidate1,pos),(candidate2,pos),...]
        candidates[key] = candidates_of_doc
    analisis_of_candidates(references, candidates)
    return data, references
