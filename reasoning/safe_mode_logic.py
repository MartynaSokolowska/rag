from config.config import NUM_CHUNKS, SAFE_MODE_CONFIG
from memory.memory import add_to_memory
from rag_query import rag_query
from reasoning.prompt import build_context_prompt, call_llm
from reasoning.validation import is_idk_answer, validate_answer
from retrieval.query_decomposition import generate_clarification_question
from utils.logger import setup_logger


def is_good_answer(answer, docs):
    if not is_idk_answer(answer) and validate_answer(answer, docs) and len(docs) != 0:
        return True
    return False


def run_safe_mode(query, docs):
    logger = setup_logger("rag-pipeline")
    for strategy in SAFE_MODE_CONFIG["retry_strategies"]:
        
        if strategy == "modify_prompt":
            logger.info("SAFE MODE | Ponowna próba ze zmianą prompta.")
            answer = retry_with_stricter_prompt(query, docs)
            if answer:
                return answer

        if strategy == "retry_retrieval":
            logger.info("SAFE MODE | Ponowna próba ze zmianą zapytania.")
            answer = retry_with_new_query(query)
            if answer:
                return answer

        if strategy == "save_to_memory":            
            logger.info("SAFE MODE | Zapis do pamięci.")
            add_to_memory(query)
            return "Nie znalazłem informacji w dokumentach."

    return "Nie wiem."


def retry_with_stricter_prompt(query, docs):
    prompt_base = """
    Odpowiedz WYŁĄCZNIE na podstawie fragmentów poniżej.
Jeśli nie znajdziesz odpowiedzi, napisz: "Nie wiem".
Dodaj cytat potwierdzający odpowiedź.
    """
    logger = setup_logger("rag-pipeline")
    prompt = build_context_prompt(query, docs, prompt_base)
    logger.info(f"SAFE MODE | Użyty prompt: {prompt}")
    answer = call_llm(prompt)
    logger.info(f"SAFE MODE | Odpowiedź po użyciu promptu: {answer}")
    if is_good_answer(answer, docs):
        logger.info(f"SAFE MODE | Odpowiedź poprawna")
        return answer

    return None

def retry_with_new_query(query):
    logger = setup_logger("rag-pipeline")
    alt_queries = generate_clarification_question(query)
    logger.info(f"SAFE MODE | Próby z nowymi zapytaniami.")
    for q in alt_queries:
        logger.info(f"SAFE MODE | Próba z zapytaniem: {q}")
        answer, used_chunks = rag_query(q)
    
        if is_good_answer(answer, used_chunks):
            logger.info(f"SAFE MODE | Odpowiedź poprawna")
            return answer

    return None

