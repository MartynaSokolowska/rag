from datetime import datetime, timedelta
import random
import requests
import json
from config.config import ES_URL, HEADERS, INDEX


def initialize_elsticsearch(data_path):
    requests.delete(f"{ES_URL}/{INDEX}", headers=HEADERS)
    body = {
        "settings": {
            "analysis": {

                "filter": {
                    "pl_morfologik": {
                        "type": "morfologik_stem"
                    }
                },

                "analyzer": {

                    "pl_lemma": {
                        "tokenizer": "standard",
                        "filter": [
                            "lowercase",
                            "pl_morfologik",  
                            "lowercase"
                        ]
                    },

                    "pl_basic": {
                        "tokenizer": "standard",
                        "filter": [
                            "lowercase"
                        ]
                    }
                }
            }
        },

        "mappings": {
            "properties": {

                "text": {
                    "type": "text",
                    "analyzer": "pl_basic"
                },

                "text_lem": {
                    "type": "text",
                    "analyzer": "pl_lemma"
                },

                "date": {
                    "type": "date"
                }
            }
        }
    }


    resp = requests.put(f"{ES_URL}/{INDEX}", headers=HEADERS, data=json.dumps(body))
    bulk_insert(generate_docs_jsonl(data_path, INDEX), INDEX) 
    return resp.json()

def generate_docs_jsonl(jsonl, index_name):
    with open(jsonl, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            random_days_ago = random.randint(0, 1095)
            fake_date = datetime.now() - timedelta(days=random_days_ago)
            yield {
                "_index": index_name,
                "_id": i,
                "_source": {
                    "text": obj["text"],
                    "text_lem": obj["text"],
                    "date": fake_date.isoformat()
                }
            }

def bulk_insert(docs, index_name):
    bulk_body = ""
    for doc in docs:
        bulk_body += json.dumps({
            "index": {
                "_index": index_name,
                "_id": doc["_id"]         
            }
        }) + "\n"
        bulk_body += json.dumps(doc["_source"]) + "\n"
    resp = requests.post(f"{ES_URL}/_bulk", headers=HEADERS, data=bulk_body)
    requests.post(f"{ES_URL}/{index_name}/_refresh", headers=HEADERS)
    return resp

#wywolanie
#bulk_insert(generate_docs_jsonl("culturax_pl_clean.jsonl", INDEX), INDEX) 


def get_bm25_es(text, k=5):
    body = {
        "query": {
            "match": {
                "text": {
                    "query": text,
                    "operator": "and"
                }
            }
        },
        "size": k,
        "_source": ["id", "text", "domain", "date"]
    }

    url = f"{ES_URL}/{INDEX}/_search"
    response = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        data=json.dumps(body)
    )
    results = response.json()

    docs = []
    for hit in results.get("hits", {}).get("hits", []):
        id = hit["_id"]
        text = hit["_source"]["text"]
        docs.append({"id": id, "text": text})

    return docs

def search_es_by_id(doc_id):
    query = {
        "query": {
            "term": {
                "_id": doc_id
            }
        }
    }
    response = requests.get(f"{ES_URL}/{INDEX}/_search", headers={"Content-Type": "application/json"}, data=json.dumps(query))
    results = response.json()

    docs = []
    for hit in results.get("hits", {}).get("hits", []):
        id = hit["_id"]
        text = hit["_source"]["text"]
        docs.append({"id": id, "text": text})

    return docs

def query_es_filter_date(text, date_from, date_to, k=100):
    body = {
        "size": k,
        "_source": ["id", "text", "domain", "date"],
        "query": {
            "bool": {
                "must": [
                    {
                        "match": {
                            "text": {
                                "query": text,
                                "operator": "and"
                            }
                        }
                    }
                ],
                "filter": [
                    {
                        "range": {
                            "date": {
                                "gte": date_from,
                                "lte": date_to
                            }
                        }
                    }
                ]
            }
        }
    }

    url = f"{ES_URL}/{INDEX}/_search"
    response = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        data=json.dumps(body)
    )

    results = response.json()

    res = []
    for hit in results.get("hits", {}).get("hits", []):
        src = hit["_source"]
        res.append({
            "id": src.get("id"),
            "text": src.get("text"),
            "domain": src.get("domain"),
            "date": src.get("date")
        })

    return res


def query_es_filter_domain(text, domain, k=100):
    body = {
        "size": k,
        "_source": ["id", "text", "domain", "date"],
        "query": {
            "bool": {
                "must": [
                    {
                        "match": {
                            "text": {
                                "query": text,
                                "operator": "and"
                            }
                        }
                    }
                ],
                "filter": [
                    {
                        "term": {
                            "domain": domain
                        }
                    }
                ]
            }
        }
    }

    url = f"{ES_URL}/{INDEX}/_search"
    response = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        data=json.dumps(body)
    )

    results = response.json()

    res = []
    for hit in results.get("hits", {}).get("hits", []):
        src = hit["_source"]
        res.append({
            "id": src.get("id"),
            "text": src.get("text"),
            "domain": src.get("domain"),
            "date": src.get("date")
        })

    return res
