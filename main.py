import os

from retrieval.elastic import initialize_elsticsearch
from retrieval.qdrant import load_data_to_qdrant
from utils.logger import setup_logger
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import rag_query
from utils.data_preparation import load_data_from_culturax
from config.config import *
from sentence_transformers import SentenceTransformer

def main():
    output_file = os.path.join(DATA_FOLDER, OUTPUT_FILE)
    model = SentenceTransformer(EMBEDDING_MODEL)

    logger = setup_logger(
        name="rag-pipeline",
        log_file="logs/" + LOG_NAME + ".log"
    )
    logger.log(f"User query: {USER_QUERY}")

    if os.path.exists(output_file):
        print(f"{output_file} already exists. Skipping download and processing.")
    else:
        print(f"{output_file} not found. Starting download and processing...")

        load_data_from_culturax(model, N_SAMPLES, MIN_LEN, MAX_LEN, output_file, NORMALIZE_EMBEDDINGS)

    resp_es = initialize_elsticsearch(output_file)
    logger.log(f"Inicjalizacja Elastic Search'a: {resp_es}")
    resp_qdrant = load_data_to_qdrant()
    logger.log(f"Inicjalizacja Qdrant'a: {resp_qdrant}")

    answer = rag_query(USER_QUERY)
    print("Odpowiedź modelu to: ", answer)
    logger.log(f"Odpowiedź modelu to: {answer}")

if __name__ == "__main__":
    main()
