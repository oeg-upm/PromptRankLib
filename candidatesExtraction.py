import spacy

def extract_candidates(input_text,key):
    keyphrase_candidate = []
    natural_language_processor = spacy.load("es_core_news_sm")  #importante
    processed_text = natural_language_processor(input_text)
    for chunk in processed_text.noun_chunks:
        keyphrase_candidate.append([chunk.text,(0,0)])
    return(keyphrase_candidate)