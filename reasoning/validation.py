import re
from reasoning.prompt import call_llm
from utils.query_utils import parse_json


def filter_retrieved(docs, query, min_tokens=30, max_docs=5):
    filtered = []
    rejected_count = 0
    
    for doc in docs:
        text = doc["text"]
        if len(filtered) >= max_docs:
            break 

        if len(text.split()) <= min_tokens:
            rejected_count += 1
            continue

        if validate_document(text, query):
            filtered.append(doc)
        else:
            rejected_count += 1
    
    return filtered, rejected_count


def validate_document(doc, query, max_retries=5):
    system = """Jesteś weryfikatorem dokumentów.

Twoim zadaniem jest ocenić, czy DOKUMENT jest semantycznie
związany z ZAPYTANIEM użytkownika w ogólnym znaczeniu
(nie musi zawierać dokładnych słów).

Jeśli dokument wnosi sensowną informację pomocną do odpowiedzi
na zapytanie → true.
Jeśli dokument jest nie na temat → false.

NIE pisz dodatkowego tekstu ani wyjaśnień.
Zwróć WYŁĄCZNIE poprawny JSON w formacie:

{
  "result": true | false
}
"""

    shots = [
        {"role": "user", "content": "PYTANIE: Jak działa RAG (Retrieval Augmented Generation)?, DOKUMENT: Retrieval Augmented Generation (RAG) is a technique that combines information retrieval with large language models to generate more accurate and grounded responses."},
        {"role": "assistant", "content": '{"result":true}'},
        
        {"role": "user", "content": "PYTANIE: Jak działa filtrowanie dokumentów w systemach RAG?, DOKUMENT: Systemy wyszukiwania wektorowego, takie jak Qdrant, umożliwiają wyszukiwanie semantyczne dokumentów na podstawie embeddingów."},
        {"role": "assistant", "content": '{"result":"true"}'},

        {"role": "user", "content": "PYTANIE: Jak działa mechanizm attention w transformerach?, DOKUMENT: Plik JSON może przechowywać dane w formacie klucz-wartość i jest często używany do konfiguracji aplikacji."},
        {"role": "assistant", "content": '{"result":"false"}'},
    ]
    
    content = f"PYTANIE: {query}, DOKUMENT: {doc}"

    for _ in range(max_retries):
        result = call_llm(content, system, shots)
        result_json = parse_json(result)
        if result_json is not None and "result" in result_json:
            return result_json["result"]
        
    return False

def validate_answer(answer, retrieved_docs, min_hits=1):
    answer = answer.lower()

    hits = 0
    for doc in retrieved_docs:
        text = (doc["text"] if isinstance(doc, dict) else doc).lower()

        for chunk in re.findall(r".{40,80}", answer):
            if chunk in text:
                hits += 1
                break

    return hits >= min_hits


def is_idk_answer(answer: str) -> bool:

    IDK_PATTERNS = [
        r"\bnie wiem\b",
        r"\bbrak (informacji|danych)\b",
        r"\bnie (mam|posiadam) (wystarczaj[aą]cych )?informacji\b",
        r"\bnie znalaz(łem|łam)\b",
        r"\bkontekst nie zawiera\b",
        r"\bnie da się (stwierdzić|określić)\b",
    ]

    if not answer:
        return True

    text = answer.lower()
    return any(re.search(p, text) for p in IDK_PATTERNS)

