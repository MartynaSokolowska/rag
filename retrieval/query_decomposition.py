from reasoning.prompt import call_llm
from utils.query_utils import parse_json


def get_es_question(user_input, max_retries=5):
    system = """Jesteś asystentem do wyszukiwania informacji.
Twoim zadaniem jest przekształcić zapytanie użytkownika, 
tak aby było dobre do wyszukiwania z użyciem ElasticSearcha 
- czyli krótkie i zawierające najważniejsze informacje.

NIE pisz dodatkowego tekstu i wyjaśnień.
Zwróć wynik WYŁĄCZNIE jako JSON w formacie:
{
"result": "..."
}
"""

    shots = [
        {"role": "user", "content": "Czym jest praca zespołowa?"},
        {"role": "assistant", "content": '{"result":"współpraca"}'},
        
        {"role": "user", "content": "PAN"},
        {"role": "assistant", "content": '{"result":"PAN"}'},

        {"role": "user", "content": "78"},
        {"role": "assistant", "content": '{"result":"78"}'},

        {"role": "user", "content": "Jak pokonać kryzys zaufania?"},
        {"role": "assistant", "content": '{"result":"kryzys zaufania"}'}
    ]
    
    for _ in range(max_retries):
        result = call_llm(user_input, system, shots)
        result_json = parse_json(result)
        if result_json is not None and "result" in result_json:
            return result_json["result"]
        
    return user_input

def get_qdrant_question(user_input, max_retries=5):
    system = """Jesteś asystentem do wyszukiwania informacji.
Twoim zadaniem jest przekształcić zapytanie użytkownika, 
tak aby było dobre do wyszukiwania z użyciem Qdrant'a 
- czyli parafrazujące i dobrze opisujące zagadnienie.

NIE pisz dodatkowego tekstu i wyjaśnień.
Zwróć wynik WYŁĄCZNIE jako JSON w formacie:
{
"result": "..."
}
"""

    shots = [
        {"role": "user", "content": "Czym jest praca zespołowa?"},
        {"role": "assistant", "content": '{"result":"Czym jest i jak wygląda praca zespołowa?"}'},
        
        {"role": "user", "content": "PAN"},
        {"role": "assistant", "content": '{"result":"Co to PAN (akronim)"}'},

        {"role": "user", "content": "Co to wolność?"},
        {"role": "assistant", "content": '{"result":"Czym jest wolność? Jakie są rodzaje wolności?"}'},

        {"role": "user", "content": "Jak pokonać kryzys zaufania?"},
        {"role": "assistant", "content": '{"result":"Jak poradzić sobie przy kryzysie zaufania?"}'}
    ]
    
    for _ in range(max_retries):
        result = call_llm(user_input, system, shots)
        result_json = parse_json(result)
        if result_json is not None and "result" in result_json:
            return result_json["result"]
        
    return user_input


def decompose_query(user_input):
    es_query = get_es_question(user_input)
    qdrant_query = get_qdrant_question(user_input)

    return {"main_question":user_input,
            "es_query":es_query,
            "qdrant_query":qdrant_query}


def generate_clarification_question(query, max_retries=5):
    system = """Jesteś asystentem do wyszukiwania informacji.
Twoim zadaniem jest zaproponować 2 interpretacje zapytania użytkownika, 
tak aby były dobre do wyszukiwania z użyciem Qdrant'a 
- czyli parafrazujące i dobrze opisujące zagadnienie.

NIE pisz dodatkowego tekstu i wyjaśnień.
Zwróć wynik WYŁĄCZNIE jako JSON w formacie:
{
"results": [
    "interpretacja 1",
    "interpretacja 2"
  ]
}
"""

    shots = [
        {"role": "user", "content": "Czym jest praca zespołowa?"},
        {"role": "assistant", "content": '{"results":["Czym jest i jak wygląda praca zespołowa?", "Kiedy ludzie współpracują, na czym polega współpraca"]}'},
        
        {"role": "user", "content": "PAN"},
        {"role": "assistant", "content": '{"results":["Czego akronimem jest PAN", "Jakie prace naukowe opiblikował PAN"]}'},

        {"role": "user", "content": "Co to wolność?"},
        {"role": "assistant", "content": '{"results":["Czym jest wolność słowa?",  "Jakie są rodzaje wolności?"]}'},

        {"role": "user", "content": "Jak pokonać kryzys zaufania?"},
        {"role": "assistant", "content": '{"results":[Jak poradzić sobie przy kryzysie zaufania?", "Co zrobić kiedy zaufanie do innej osoby się pogarsza?"]}'}
    ]
    
    for _ in range(max_retries):
        result = call_llm(query, system, shots)
        result_json = parse_json(result)
        if result_json is not None and "results" in result_json:
            return result_json["results"]
        
    return [query]


