import os

from retrieval.elastic import initialize_elsticsearch
from retrieval.qdrant import load_data_to_qdrant
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import rag_query
from utils.data_preparation import load_data_from_culturax
from config.config import *
from sentence_transformers import SentenceTransformer

def main():
    output_file = os.path.join(DATA_FOLDER, OUTPUT_FILE)
    model = SentenceTransformer(EMBEDDING_MODEL)

    if os.path.exists(output_file):
        print(f"{output_file} already exists. Skipping download and processing.")
    else:
        print(f"{output_file} not found. Starting download and processing...")

        load_data_from_culturax(model, N_SAMPLES, MIN_LEN, MAX_LEN, output_file, NORMALIZE_EMBEDDINGS)

    resp_es = initialize_elsticsearch(output_file)
    resp_qdrant = load_data_to_qdrant()
    # to samo dla qdranta napisać :)

    answer = rag_query(USER_QUERY)
    print("Odpowiedź modelu to: ", answer)

if __name__ == "__main__":
    main()
