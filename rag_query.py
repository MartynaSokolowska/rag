from memory.memory import add_to_memory
from reasoning.chunking import *
from reasoning.prompt import build_context_prompt, call_llm
from reasoning.safe_mode_logic import is_good_answer, run_safe_mode
from reasoning.validation import filter_retrieved, is_idk_answer, validate_answer
from retrieval.query_decomposition import decompose_query
from retrieval.fusion import retrive
from config.config import CHUNK_TYPE, CHUNK_SIZE, NUM_CHUNKS, SAFE_MODE_CONFIG


def get_answer_for_query(query):
    base_queries = decompose_query(query)

    retrived = retrive(base_queries)
    selected_docs, rejected_count = filter_retrieved(retrived, query, max_docs=NUM_CHUNKS)

    chunks = []
    for doc in selected_docs:
        if CHUNK_TYPE == "TOKENS":
            chunks.extend(chunk_by_chars(doc["text"], CHUNK_SIZE))
        else:
            chunks.extend(chunk_by_tokens(doc["text"], CHUNK_SIZE))

    prompt = build_context_prompt(query, chunks[:NUM_CHUNKS])
    answer = call_llm(prompt)
    return answer, chunks


def rag_query(user_input):

    answer, used_chunks = get_answer_for_query(user_input)

    if not SAFE_MODE_CONFIG["enabled"]:
        if is_idk_answer(answer):
            add_to_memory(user_input)
            return "Nie wiem."

        if not validate_answer(answer, used_chunks):
            add_to_memory(user_input)
            return "Nie mam wystarczających danych."
        
        if len(used_chunks) == 0:
            add_to_memory(user_input)
            return "Nie znalazłem informacji."
        
        return answer
    
    elif not is_good_answer(answer, used_chunks):
        run_safe_mode(user_input, used_chunks)
    
    return answer