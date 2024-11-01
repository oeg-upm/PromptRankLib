import spacy
from spacy.tokens import Doc
from typing import Generator
from spacy.tokens import Span


class candidatesExtraction:

    def __init__(self) -> None:
        self.natural_language_processor = spacy.load("es_core_news_sm")  #importante
    
    def merge_adp_chunks(self, chunk: Span, current_chunk_completed = '') -> None:
            #TODO: EL START Y EL END NO COINCIDEN BIEN CUANDO AÑADIMOS CHUNK de más de una llamada
            current_chunk_completed = current_chunk_completed + chunk.text
            start_of_next_chunk = chunk.end + 1
            chunk = next(self.chunk_generator, None)
            if chunk != None:
                if chunk.start == start_of_next_chunk:
                    token_after_chunk_label = self.processed_text[chunk.end].text
                    if token_after_chunk_label == 'de':
                        current_chunk_completed = current_chunk_completed + ' ' + self.processed_text[chunk.start - 1].text + ' ' 
                        self.merge_adp_chunks(chunk,    current_chunk_completed)
                    else: 
                        current_chunk_completed = current_chunk_completed + ' ' + self.processed_text[chunk.start - 1].text + ' ' + chunk.text
                        add_text = self.clean_determinants(current_chunk_completed)
                        self.keyphrase_candidate.append([add_text, (chunk.end-len(current_chunk_completed.split()), chunk.end)])
                        return
                else:
                    add_text = self.clean_determinants(current_chunk_completed)
                    self.keyphrase_candidate.append([add_text, (chunk.end-len(current_chunk_completed.split()), chunk.end)])
                    token_after_chunk = self.processed_text[chunk.end]
                    if token_after_chunk.text == 'de':
                        self.merge_adp_chunks(chunk)
                    else:
                        add_text = self.clean_determinants(chunk.text)
                        self.keyphrase_candidate.append([add_text, (chunk.start, chunk.end)])
                        return
    
    def clean_determinants(self, text_to_clean: str) -> str:
        processor = self.natural_language_processor(text_to_clean)
        if(processor[0].pos_ == 'DET' and len(processor) > 1):
            text_to_clean = text_to_clean.split(' ', 1)[1]
        return text_to_clean

    def extract_candidates(self, input_text: str, key: str):
        self.keyphrase_candidate = []
        self.processed_text = self.natural_language_processor(input_text)
        self.chunk_generator = self.processed_text.noun_chunks
        for chunk in self.chunk_generator:
            if chunk.end < len(self.processed_text):
                token_after_chunk = self.processed_text[chunk.end]
                if token_after_chunk.text == 'de':
                    self.merge_adp_chunks(chunk)
            else:
                add_text = self.clean_determinants(chunk.text)
                self.keyphrase_candidate.append([add_text, (chunk.start, chunk.end)])
        # TODO: recorrer todos los candidatos y con spacy quitar el articulo determinados y indeterminados del principio de los candidatos
        return(self.keyphrase_candidate)
    
# este -> pos_ : DET
