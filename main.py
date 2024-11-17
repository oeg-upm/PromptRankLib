import logging
import datetime
import argparse

from processDataset import clean_dataset
from torch.utils.data import DataLoader

from keyphraseExtraction import keyphrase_selection

def get_setting_dict(encoder_header: str, prompt: str, max_len: int, model_version: str):
    setting_dict = {}
    setting_dict["max_len"] = max_len
    setting_dict["temp_en"] = encoder_header
    setting_dict["temp_de"] = prompt
    setting_dict["model"] = model_version
    #TODO: what are the advantages of enable_filter and enable_pos
    setting_dict["enable_filter"] = False
    setting_dict["enable_pos"] = True
    setting_dict["position_factor"] = 1.2e8
    setting_dict["length_factor"] = 0.6
    return setting_dict

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--regular_expresion', dest='regular_expresion_value', action='store_true',
                    help='Set the regular_expresion value to True.')
    parser.add_argument('--no-regular_expresion', dest='regular_expresion_value', action='store_false',
                    help='Set the regular_expresion value to False.')
    parser.set_defaults(regular_expresion_value=True)
    parser.add_argument("--greedy",
                        default="FIRST",
                        type=str,
                        required=False,
                        help="Method to be used while extracting candidates with regular expresion. LONGEST/FIRST/COMBINED/NONE(we will get all coincidences)")
    parser.add_argument("--title_graph_candidates_extraction",
                        default="Extracción Candidatos",
                        type=str,
                        required=False,
                        help="Title for the grafic")
    parser.add_argument("--batch_size",
                        default=128,
                        type=int,
                        required=False,
                        help="Batch size para evaluar el modelo")
    parser.add_argument("--encoder_header",
                        default="Texto:",
                        type= str,
                        required=False,
                        help= "The text that is going to precede the input at the encoder")
    parser.add_argument("--prompt",
                        default="Este texto habla principalmente de ",
                        type= str,
                        help= "The prompt that will precede the candidate")
    parser.add_argument("--max_len",
                        default= 512,
                        type= int,
                        help= "Max length that the tokenizer will support for encoding the text")
    parser.add_argument("--model_version",
                        default= "base",
                        type= str,
                        help= "The version of MT5 moder to be used")
    args = parser.parse_args()
    return args

def main():
    args = parse_argument()
    logger = logging.getLogger(__name__)
    setting_dict = get_setting_dict(args.encoder_header, args.prompt, args.max_len, args.model_version)
    logging.basicConfig(filename='PromptRankLib.log', encoding='utf-8', filemode='w', level=logging.DEBUG)
    logger.info(f"The main program has started at {datetime.datetime.now()}")
    # TODO: PASAR EL LOGGER A CLEAN DATASET PARA IR HACIENDO UN RASTREO DE LA EJECUCIÓN
    dataset, documents_list, labels, labels_stemed = clean_dataset(args.regular_expresion_value, args.title_graph_candidates_extraction, args.greedy,
                                                                   args.encoder_header, args.prompt, args.max_len, args.model_version)
    dataloader = DataLoader(dataset, num_workers=4, batch_size=args.batch_size)
    keyphrase_selection(setting_dict, documents_list, labels_stemed, labels, dataloader, logger, args.model_version)
    logger.info(f'The execution has finished {datetime.datetime.now()}')

if __name__ == "__main__":
    main()