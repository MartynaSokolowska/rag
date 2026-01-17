# USER QUERY
USER_QUERY = "Jaki można poczytać ciekawe ksiązki?"

# === DATA SETTINGS ===
DATA_FOLDER = "data"
OUTPUT_FILE = "culturax_pl_clean.jsonl"  
N_SAMPLES = 50_000
MIN_LEN = 500
MAX_LEN = 550

# === MODEL SETTINGS ===
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
NORMALIZE_EMBEDDINGS = True

# === LLM & PROMPT SETTINGS ===
LLM_MODEL = 'gemma2:2b'
TASK_PROMPT = "Odpowiedz na pytanie na podstawie poniższego kontekstu.Jeśli nie ma odpowiedzi w kontekście, napisz \"Nie wiem\"."

# === CHUNKING SETTINGS ===
CHUNK_TYPE = "TOKENS"  # works also with CHARS
CHUNK_SIZE = 250
NUM_CHUNKS = 3

# === ES SETTINGS ===
ES_URL = "http://localhost:9200"
HEADERS = {
    "Content-Type": "application/vnd.elasticsearch+json; compatible-with=8",
    "Accept": "application/vnd.elasticsearch+json; compatible-with=8"
}
INDEX = "culturax_lab8"

# === MODE SETTINGS ===
SAFE_MODE_CONFIG = {
    "enabled": True,
    "retry_strategies": [
        "modify_prompt",
        "retry_retrieval",
        "save_to_memory"
    ],
    "max_retries": 2
}
