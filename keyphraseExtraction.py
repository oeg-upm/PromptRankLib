from logging import Logger
import numpy as np
import pandas as pd

from transformers import MT5ForConditionalGeneration
from transformers import MT5Tokenizer

import torch
from torch import device as torchDevice
from torch.utils.data.dataloader import DataLoader

from tqdm import tqdm

def init(setting_dict: list, model_version: str) -> None:
    '''
    Init template, max length and tokenizer.
    '''
    global MAX_LEN, temp_en, temp_de, tokenizer, enable_pos, length_factor, position_factor
    MAX_LEN = setting_dict["max_len"]
    temp_en = setting_dict["temp_en"]
    temp_de = setting_dict["temp_de"]
    enable_pos = setting_dict["enable_pos"]
    position_factor = setting_dict["position_factor"]
    length_factor = setting_dict["length_factor"]

    tokenizer = MT5Tokenizer.from_pretrained(f"google/mt5-{model_version}", model_max_length=MAX_LEN)

def keyphrase_selection(setting_dict: list, documents_list: list, labels_stemed: list,
                         labels: list, dataloader: DataLoader, logger: Logger, model_version: str) -> None:
    init(setting_dict, model_version)
    device = torchDevice("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MT5ForConditionalGeneration.from_pretrained(f"google/mt5-{model_version}")
    model.to(device)
    model.eval()
    cos_similarity_list = {}
    doc_id_list = []
    candidate_list = []
    cos_score_list = []
    pos_list = []
    template_len = tokenizer(temp_de, return_tensors="pt")["input_ids"].shape[1] - 3    #TODO: investigate wy we subtract 3 from the length
    for id, [en_input_ids,  en_input_mask, de_input_ids, dic] in enumerate(tqdm(dataloader,desc="Evaluating:")):    # dic = {"de_input_len":de_input_len, "candidate":candidate, "idx":idx, "pos":can_and_pos[1][0]}
        en_input_ids = en_input_ids.to(device)
        en_input_mask = en_input_mask.to(device)
        de_input_ids = de_input_ids.to(device)
        score = np.zeros(en_input_ids.shape[0])
        with torch.no_grad():   # disabling gradient calculation
            output = model(input_ids=en_input_ids, attention_mask=en_input_mask, decoder_input_ids=de_input_ids)[0]
            for i in range(template_len, de_input_ids.shape[1] - 3):    # Range between the end of the prompt and the final of the candidate 
                logits = output[:, i, :]    # selects the logits for all batches at position i in the sequence
                logits = logits.softmax(dim=1)  # each candidate probability is also represented by a array length 32128
                logits = logits.cpu().numpy()   # convert pythorch tensor into numpy array, this can only be done in the cpu
                for j in range(de_input_ids.shape[0]):  # # j refers to each prompt+candidate input (index)
                    if i < dic["de_input_len"][j]:
                        score[j] = score[j] + np.log(logits[j, int(de_input_ids[j][i + 1])])    # to select corresponding tensor in vector score
                    elif i == dic["de_input_len"][j]:
                        score[j] = score[j] / np.power(dic["de_input_len"][j] - template_len, length_factor)    # calculate the penalty for candidate lenght
            doc_id_list.extend(dic["idx"])
            candidate_list.extend(dic["candidate"])
            cos_score_list.extend(score)
            pos_list.extend(dic["pos"])
    cos_similarity_list["doc_id"] = doc_id_list
    cos_similarity_list["candidate"] = candidate_list
    cos_similarity_list["score"] = cos_score_list
    cos_similarity_list["pos"] = pos_list
    cosine_similarity_rank = pd.DataFrame(cos_similarity_list)
    for i in range(len(documents_list)):
        doc_len = len(documents_list[i].split())
        doc_results = cosine_similarity_rank.loc[cosine_similarity_rank['doc_id']==i]
        ranked_keyphrases = doc_results.sort_values(by='score', ascending=False)
        top_k = ranked_keyphrases.reset_index(drop = True)  # reseting the index and 
        top_k_can = top_k.loc[:, ['candidate']].values.tolist() # producing a list with orderer candidates
        candidates_set = set()  # for query for exiting values in an easier way
        candidates_dedup = []   # for selecting top_k candidates
        for temp in top_k_can:
            temp = temp[0].lower()
            if temp in candidates_set:
                continue
            else:
                candidates_set.add(temp)
                candidates_dedup.append(temp)
        j = 0
        Matched = candidates_dedup[:15]
        number_matches_15_first_candidates = 0
        #TODO: need to add porter = PorterStemmer()?
        for id, temp in enumerate(candidates_dedup[0:15]):
            if (temp in labels[i]):
                Matched[id] = [temp]
                number_matches_15_first_candidates += 1
            j += 1
        logger.info("TOP-K {}: {} \n".format(i, Matched))   
        logger.info("Reference {}: {} \n".format(i,labels[i]))
    pass