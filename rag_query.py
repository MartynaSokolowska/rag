from memory.memory import add_to_memory
from reasoning.chunking import *
from reasoning.prompt import build_context_prompt, call_llm
from reasoning.validation import filter_retrieved, is_idk_answer, validate_answer
from retrieval.query_decomposition import decompose_query
from retrieval.fusion import retrive
from config.config import CHUNK_TYPE, CHUNK_SIZE, NUM_CHUNKS, SAFE_MODE_CONFIG
from utils.logger import setup_logger


def rag_query(query):
    logger = setup_logger("rag-pipeline")
    base_queries = decompose_query(query)

    logger.info(f"Zdekomponowane zapytanie: {base_queries}")

    retrived = retrive(base_queries)
    selected_docs, rejected_count = filter_retrieved(retrived, query, max_docs=NUM_CHUNKS)

    logger.info(f"Liczba odrzuconych dokumentów: {rejected_count}")

    chunks = []
    for doc in selected_docs:
        if CHUNK_TYPE == "TOKENS":
            chunks.extend(chunk_by_chars(doc["text"], CHUNK_SIZE))
        else:
            chunks.extend(chunk_by_tokens(doc["text"], CHUNK_SIZE))

    prompt = build_context_prompt(query, chunks[:NUM_CHUNKS])
    logger.info(f"Used prompt {prompt}")
    answer = call_llm(prompt)
    logger.info(f"Odpowiedź modelu przed sprawdzeniem poprawności: {answer}")
    return answer, chunks[:NUM_CHUNKS]
