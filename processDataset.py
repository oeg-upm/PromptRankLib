import codecs
import os
import re
from transformers import MT5Tokenizer
from transformers.models.t5.tokenization_t5 import T5Tokenizer
from torch.utils.data import Dataset
from tqdm import tqdm

import itertools
import spacy

from analisisCandidates import analisis_of_candidates
from candidatesExtraction import candidatesExtraction

MAX_LEN_DOCUMENT = 512
TEMPLATE_ENCODER = "Texto:"
TEMPLATE_DECODER = "Este texto habla principalmente de "

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
    return text

class KeyPhrasesExtractionDataset(Dataset):
    def __init__(self, docs_pairs):

        self.docs_pairs = docs_pairs
        self.total_examples = len(self.docs_pairs)

    def __len__(self):
        return self.total_examples

    def __getitem__(self, idx):

        doc_pair = self.docs_pairs[idx]
        en_input_ids = doc_pair[0][0]
        en_input_mask = doc_pair[1][0]
        de_input_ids = doc_pair[2][0]
        dic = doc_pair[3]

        return [en_input_ids, en_input_mask, de_input_ids, dic]

def generate_doc_pairs(doc: str, candidates: list, idx: int, tokenizer: T5Tokenizer) -> list[list,int]:
    candidate_document_pairs = []
    inputs_ids = tokenizer(doc, max_length=MAX_LEN_DOCUMENT, padding="max_length", truncation=True, return_tensors="pt")
    enconder_input_ids = inputs_ids["input_ids"]
    enconder_input_mask = inputs_ids["attention_mask"]
    for candidate_and_pos in candidates:
        candidate = candidate_and_pos[0]
        decoder_input = TEMPLATE_DECODER + candidate + " ."
        decoder_input_ids = tokenizer(decoder_input, max_length=30, padding="max_length", truncation=True, return_tensors="pt")["input_ids"]
        decoder_input_ids [0, 0] = 0
        de_input_len = (decoder_input_ids[0] == tokenizer.eos_token_id).nonzero()[0].item() - 2
        dic = {"de_input_len":de_input_len, "candidate":candidate, "idx":idx, "pos":candidate_and_pos[1][0]}
        candidate_document_pairs.append([enconder_input_ids, enconder_input_mask, decoder_input_ids, dic])
    return candidate_document_pairs, 0      #TODO: implement the count value
    
def clean_dataset( regular_expression : bool, graph_title: str, greedy: str, data_path="data/docsutf8", labels_path="data/keys"):
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
    #TODO: remove 2 lines above for complete extraction
    data = dict(itertools.islice(data.items(), 3)) 
    references = dict(itertools.islice(references.items(), 3))
    candidates_extractor = candidatesExtraction(regular_expression,greedy)
    document_pairs = []
    documents_list = []
    labels = []
    tokenizer = MT5Tokenizer.from_pretrained("google/mt5-base")
    for idx, (key, doc) in tqdm(enumerate(data.items()),desc="Calculating the candidates of each document",total=len(data)):
        labels.append([ref.replace(" \n", "") for ref in references[key]])
        candidates_of_each_doc = candidates_extractor.extract_candidates(doc, key)    # get the candidates with FORMAT [(candidate1,pos),(candidate2,pos),...]
        candidates[key] = candidates_of_each_doc    # get the candidates with FORMAT [(candidate1,pos),(candidate2,pos),...]
        doc = ' '.join(doc.split()[:MAX_LEN_DOCUMENT])
        documents_list.append(doc)
        doc = TEMPLATE_ENCODER + "\"" + doc + "\""
        candidate_document_input, count = generate_doc_pairs(doc, candidates_of_each_doc, idx, tokenizer)
        document_pairs.extend(candidate_document_input)
    #analisis_of_candidates(references, candidates, title = graph_title)
    dataset = KeyPhrasesExtractionDataset(document_pairs)
    print(f'The number of candidates is {dataset.total_examples}')
    labels_stemed = []  #TODO: implement labels setemed
    return dataset, documents_list, labels, labels_stemed