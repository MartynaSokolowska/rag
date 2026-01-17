import os
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

    answer = rag_query(USER_QUERY)
    print("Odpowiedź modelu to: ", answer)

if __name__ == "__main__":
    main()
