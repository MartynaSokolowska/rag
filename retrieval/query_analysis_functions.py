from config.config import N_SAMPLES
from reasoning.prompt import call_llm
from utils.query_utils import parse_json, validate_date_filter

def is_id(user_input):
    user_input = user_input.strip()

    if not user_input.isdigit():
        return False

    value = int(user_input)
    return 1 <= value <= N_SAMPLES


def is_semantic(user_input, max_retries=5):
    system = """You are a query classifier.

Your task is to decide whether a user query is SEMANTIC or NOT_SEMANTIC.

A SEMANTIC query:
- asks about meaning, concepts, explanations, causes, effects, opinions, summaries
- benefits from semantic similarity search (embeddings, vector databases)

A NOT_SEMANTIC query:
- is purely factual with a single exact answer
- is a command, a greeting, or a navigation request
- refers to exact values, IDs, filenames, or specific records

Return ONLY valid JSON in exactly this format:

{
  "semantic": true | false
}

Do NOT add any explanations or extra text.
"""

    shots = [
        {"role": "user", "content": "Czym jest praca zespołowa?"},
        {"role": "assistant", "content": '{"semantic":true}'},
        
        {"role": "user", "content": "PAN"},
        {"role": "assistant", "content": '{"semantic":false}'},

        {"role": "user", "content": "id: 78"},
        {"role": "assistant", "content": '{"semantic":false}'},

        {"role": "user", "content": "Jak pokonać kryzys zaufania?"},
        {"role": "assistant", "content": '{"semantic":true}'}
    ]
    
    for _ in range(max_retries):
        result = call_llm(user_input, system, shots)
        result_json = parse_json(result)
        if result_json is not None and "semantic" in result_json:
            return result_json["semantic"]
        
    return True


def date_filter(user_input, max_retries=5):
    system = """You are a date filter extraction assistant.

Your task is to determine whether a user query contains a time or date constraint
that can be used as a search filter.

If a date filter CAN be applied:
- set "date_filter" to true
- extract "date_from" and/or "date_to" in ISO 8601 format (YYYY-MM-DD)
- if only one bound is implied, set the other to null

If a date filter CANNOT be applied:
- set "date_filter" to false
- set both "date_from" and "date_to" to null

Rules:
- Use best reasonable interpretation of relative dates (e.g. "last year", "recent")
- If the date range is vague or ambiguous but still usable, prefer a conservative range
- NEVER guess exact dates if no temporal reference is present

Return ONLY valid JSON in exactly this format:

{
  "date_filter": true | false,
  "date_from": "YYYY-MM-DD" | null,
  "date_to": "YYYY-MM-DD" | null
}

Do NOT include explanations or any additional text.

"""

    shots = [
        {"role": "user", "content": "Czym jest praca zespołowa?"},
        {"role": "assistant", "content": '{"date_filter":false, "date_from": null, "date_to": null}'},
        
        {"role": "user", "content": "Articles about AI from 2020 to 2022"},
        {"role": "assistant", "content": '{"date_filter": true, "date_from": "2020-01-01", "date_to": "2022-12-31"}'},

        {"role": "user", "content": "News about elections last year"},
        {"role": "assistant", "content": '{"date_filter": true, "date_from": "2025-01-01", "date_to": "2025-12-31"}'},

        {"role": "user", "content": "Research papers after 2018"},
        {"role": "assistant", "content": '{"date_filter": true, "date_from": "2019-01-01", "date_to":null}'}
    ]
    
    for _ in range(max_retries):
        result = call_llm(user_input, system, shots)
        result_json = parse_json(result)
        if result_json is not None and "date_filter" in result_json:
            if validate_date_filter(result_json):
                return result_json
        
    return {"date_filter":False}


def domain_filter(user_input, max_retries=5):
    system = """You are a domain filter extraction assistant.

Your task is to determine whether a user query contains a filterable domain.
The only valid domains are:

- mC4
- OSCAR-2109
- OSCAR-2301
- OSCAR-2201

If the query contains a reference to one of these domains:
- set "domain_filter" to true
- set "domain" to the domain name (string)

If no valid domain is mentioned:
- set "domain_filter" to false
- set "domain" to null

Return ONLY valid JSON in exactly this format:

{
  "domain_filter": true | false,
  "domain": "DOMAIN_NAME" | null
}

Do NOT add explanations, comments, or any extra text.
"""

    shots = [
        {"role": "user", "content": "Search articles from mC4 about AI"},
        {"role": "assistant", "content": '{"domain_filter": true, "domain": "mC4"}'},
        
        {"role": "user", "content": "News from OSCAR-2301 about elections"},
        {"role": "assistant", "content": '{"domain_filter": true, "domain": "OSCAR-2301"}'},

        {"role": "user", "content": "Articles from Wikipedia"},
        {"role": "assistant", "content": '{"domain_filter": false, "domain": null}'},

        {"role": "user", "content": "Jak pokonać kryzys zaufania?"},
        {"role": "assistant", "content": '{"domain_filter": false, "domain": null}'}
    ]
    
    for _ in range(max_retries):
        result = call_llm(user_input, system, shots)
        result_json = parse_json(result)
        if result_json is not None and "domain_filter" in result_json:
            if not result_json["domain_filter"] or (result_json["domain_filter"] and result_json["domain"] in ["mC4", "OSCAR-2109", "OSCAR-2301", "OSCAR-2201"]):
                return result_json
        
    return {"domain_filter":False}
