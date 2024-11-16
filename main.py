import logging
import datetime
import argparse

from processDataset import clean_dataset
from torch.utils.data import DataLoader

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
                        help="Título gráfico análisis extracción candidatos")
    parser.add_argument("--batch_size",
                        default=128,
                        type=int,
                        required=False,
                        help="Batch size para evaluar el modelo")
    args = parser.parse_args()
    return args

def main():
    args = parse_argument()
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='PromptRankLib.log', encoding='utf-8', filemode='w', level=logging.DEBUG)
    logger.info(f"The main program has started at {datetime.datetime.now()}")
    # TODO: PASAR EL LOGGER A CLEAN DATASET PARA IR HACIENDO UN RASTREO DE LA EJECUCIÓN
    dataset, documents_list, labels, labels_stemed = clean_dataset(args.regular_expresion_value, args.title_graph_candidates_extraction, args.greedy)
    dataloader = DataLoader(dataset, num_workers=4, batch_size=args.batch_size)

if __name__ == "__main__":
    main()