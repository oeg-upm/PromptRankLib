import logging
import datetime

from cleanDataset import clean_dataset

def parse_argument():...

def main():
    args = parse_argument()
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='PromptRankLib.log', encoding='utf-8', filemode='w', level=logging.DEBUG)
    logger.info(f"The main program has started at {datetime.datetime.now()}")
    # TODO: PASAR EL LOGGER A CLEAN DATASET PARA IR HACIENDO UN RASTREO DE LA EJECUCIÓN
    data, labels = clean_dataset()

if __name__ == "__main__":
    main()