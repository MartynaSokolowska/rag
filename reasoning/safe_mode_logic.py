from config.config import NUM_CHUNKS, SAFE_MODE_CONFIG
from memory.memory import add_to_memory
from rag_query import get_answer_for_query
from reasoning.prompt import build_context_prompt, call_llm
from reasoning.validation import is_idk_answer, validate_answer
from retrieval.query_decomposition import generate_clarification_question


def is_good_answer(answer, docs):
    if not is_idk_answer(answer) and validate_answer(answer, docs) and len(docs) != 0:
        return True
    return False


def run_safe_mode(query, docs):
    for strategy in SAFE_MODE_CONFIG["retry_strategies"]:
        
        if strategy == "modify_prompt":
            answer = retry_with_stricter_prompt(query, docs)
            if answer:
                return answer

        if strategy == "retry_retrieval":
            answer = retry_with_new_query(query, docs)
            if answer:
                return answer

        if strategy == "save_to_memory":
            add_to_memory(query, reason="safe_mode_failed")
            return "Nie znalazłem informacji w dokumentach."

    return "Nie wiem."


def retry_with_stricter_prompt(query, docs):
    prompt_base = """
    Odpowiedz WYŁĄCZNIE na podstawie fragmentów poniżej.
Jeśli nie znajdziesz odpowiedzi, napisz: "Nie wiem".
Dodaj cytat potwierdzający odpowiedź.
    """

    prompt = build_context_prompt(query, docs, prompt_base)
    answer = call_llm(prompt)

    if is_good_answer(answer, docs):
        return answer

    return None

def retry_with_new_query(query, chunks):
    alt_queries = generate_clarification_question(query)

    for q in alt_queries:
        answer, used_chunks = get_answer_for_query(q)

        if is_good_answer(answer, used_chunks):
            return answer

    return None

