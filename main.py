import logging
import datetime
import argparse

from cleanDataset import clean_dataset

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--regular_expresion', dest='regular_expresion_value', action='store_true',
                    help='Set the regular_expresion value to True.')
    parser.add_argument('--no-regular_expresion', dest='regular_expresion_value', action='store_false',
                    help='Set the regular_expresion value to False.')
    parser.set_defaults(regular_expresion_value=True)
    parser.add_argument("--greedy",
                        default="LONGEST",
                        type=str,
                        required=False,
                        help="Method to be used while extracting candidates with regular expresion")
    parser.add_argument("--title_graph_candidates_extraction",
                        default="Extracción Candidatos",
                        type=str,
                        required=False,
                        help="Título gráfico análisis extracción candidatos")
    args = parser.parse_args()
    return args

def main():
    args = parse_argument()
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='PromptRankLib.log', encoding='utf-8', filemode='w', level=logging.DEBUG)
    logger.info(f"The main program has started at {datetime.datetime.now()}")
    # TODO: PASAR EL LOGGER A CLEAN DATASET PARA IR HACIENDO UN RASTREO DE LA EJECUCIÓN
    data, labels = clean_dataset(args.regular_expresion_value, args.title_graph_candidates_extraction, args.greedy)

if __name__ == "__main__":
    main()