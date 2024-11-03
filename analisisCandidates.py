import matplotlib.pyplot as plt
import numpy as np

def appears_in_candidates(document_key: str, candidates: dict, keyphrase_to_search: str)-> bool:
    list_of_candidates = candidates.get(document_key)
    for candidate_tuple in list_of_candidates:
        candidate = candidate_tuple[0]
        if candidate == keyphrase_to_search:
            return True
    return False

def analisis_of_candidates (references : dict, candidates : dict, title: str)-> None:
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
    
    mean_value = np.mean(percentaje_correct)
    plt.plot(percentaje_correct)
    plt.axhline(y=mean_value, color='r', linestyle='-', label='Mean')
    plt.text(x=len(percentaje_correct)-1, y=mean_value, s=f'       {mean_value:.2f}', color='red', va='center')
    plt.legend()
    plt.title(title)
    plt.xticks([])  # hiding x axis numbers
    plt.ylabel('Porcentaje de Frases Clave Detectadas')
    plt.savefig(title)