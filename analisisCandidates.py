import matplotlib.pyplot as plt
import numpy as np

def appears_in_candidates(document_key: str, candidates: dict, keyphrase_to_search: str)-> bool:
    list_of_candidates = candidates.get(document_key)
    for candidate_tuple in list_of_candidates:
        candidate = candidate_tuple[0]
        if candidate == keyphrase_to_search:
            return True
    return False

def analisis_of_candidates (references : dict, candidates : dict)-> None:
    percentaje_correct = []
    for idx, (key, array_of_keyphrases) in enumerate(references.items()):
        number_of_keyphrases = 0
        detected = 0
        for keyphrase in array_of_keyphrases:
            appears = appears_in_candidates(key, candidates, keyphrase)
            if appears:
                detected =  detected + 1
            number_of_keyphrases = number_of_keyphrases + 1
        percentaje_correct.append((detected/number_of_keyphrases)*100)
    
    plt.plot(percentaje_correct)
    plt.axhline(y=np.mean(percentaje_correct), color='r', linestyle='-', label='Mean')
    plt.legend()
    plt.ylabel('Detected keyphrases')
    plt.savefig('KFdetected')