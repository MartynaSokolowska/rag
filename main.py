import os

from memory.memory import add_to_memory
from reasoning.safe_mode_logic import is_good_answer, run_safe_mode
from reasoning.validation import is_idk_answer, validate_answer
from retrieval.elastic import initialize_elsticsearch
from retrieval.qdrant import load_data_to_qdrant
from utils.logger import setup_logger
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from rag_query import rag_query
from utils.data_preparation import load_data_from_culturax
from config.config import *
from sentence_transformers import SentenceTransformer

def main(user_input):
    output_file = os.path.join(DATA_FOLDER, OUTPUT_FILE)
    model = SentenceTransformer(EMBEDDING_MODEL)

    logger = setup_logger(
        name="rag-pipeline",
        log_file="logs/" + LOG_NAME + ".log"
    )
    logger.info(f"User query: {user_input}")

    if os.path.exists(output_file):
        print(f"{output_file} already exists. Skipping download and processing.")
    else:
        print(f"{output_file} not found. Starting download and processing...")

        load_data_from_culturax(model, N_SAMPLES, MIN_LEN, MAX_LEN, output_file, NORMALIZE_EMBEDDINGS)

    if INITIALIZE:
        resp_es = initialize_elsticsearch(output_file)
        logger.info(f"Inicjalizacja Elastic Search'a: {resp_es}")
        resp_qdrant = load_data_to_qdrant(output_file)
        logger.info(f"Inicjalizacja Qdrant'a: {resp_qdrant}")
    else:
        logger.info("Skipping ES and Qdrant initialization")

    answer, used_chunks = rag_query(user_input)
    #print("Odpowiedź modelu to: ", answer)
    logger.info(f"Odpowiedź modelu to: {answer}")

    if not SAFE_MODE_CONFIG["enabled"]:
        if is_idk_answer(answer):
            add_to_memory(user_input)
            logger.info("Zapytanie dodane do pamięci - model nie zna odpowiedzi.")
            return "Nie wiem."

        if not validate_answer(answer, used_chunks):
            add_to_memory(user_input)
            logger.info("Zapytanie dodane do pamięci - odpowiedź nie zawiera cytatu.")
            return "Nie mam wystarczających danych."
        
        if len(used_chunks) == 0:
            add_to_memory(user_input)
            logger.info("Zapytanie dodane do pamięci - nie ma żadnego znalezionego dokumentu.")
            return "Nie znalazłem informacji."
        
        return answer
    
    elif not is_good_answer(answer, used_chunks):
        run_safe_mode(user_input, used_chunks)

if __name__ == "__main__":
    main(USER_QUERY)
