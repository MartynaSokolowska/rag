from retrieval.elastic import get_bm25_es, query_es_filter_date, query_es_filter_domain, search_es_by_id
from retrieval.qdrant import get_knn_qdrant, query_qdrant_filter_date, query_qdrant_filter_domain
from retrieval.query_analysis_functions import *
from utils.logger import setup_logger


def rrf_fusion(results_lists, weights=None, k=60):
    if weights is None:
        weights = [1.0] * len(results_lists)
    assert len(weights) == len(results_lists)

    scores = {}

    for i, results in enumerate(results_lists):
        w = weights[i]
        for rank, doc in enumerate(results):
            doc_id = doc["id"]
            scores.setdefault(doc_id, {
                "score": 0,
                "text": doc["text"]
            })
            scores[doc_id]["score"] += w / (k + rank + 1)

    return sorted(scores.values(), key=lambda x: x["score"], reverse=True)


def retrive(base_queries):
    logger = setup_logger("rag-pipeline")
    user_input = base_queries["main_query"]
    es_query = base_queries["es_query"]
    qdrant_query = base_queries["qdrant_query"]

    es_docs = None
    qdrant_docs = None

    if is_id(user_input):
        logger.info("RETRIVAL | Searching by id")
        return search_es_by_id(user_input)
    
    domain_ = domain_filter(user_input)
    if domain_["domain_filter"]:
        logger.info("RETRIVAL | Using domain filter")
        domain = domain_["domain"]
        es_docs = query_es_filter_domain(es_query, domain)
        qdrant_docs = query_qdrant_filter_domain(qdrant_query, domain)
    
    date_ = date_filter(user_input)
    if date_["date_filter"]:
        logger.info("RETRIVAL | Using date filter")
        date_from = date_["date_from"]
        date_to = date_["date_to"]
        es_docs = query_es_filter_date(es_query, date_from, date_to)
        qdrant_docs = query_qdrant_filter_date(qdrant_query, date_from, date_to)
    
    if es_docs is None:
        es_docs = get_bm25_es(es_query, 20)
        qdrant_docs = get_knn_qdrant(qdrant_query, 20)

    if is_semantic(user_input):
        weights = [0.4, 0.6]
        logger.info(f"RETRIVAL | Query is semantic - weights (ES, Qdrant) {weights}")
    else:
        weights = [0.8, 0.2]
        logger.info(f"RETRIVAL | Query is not semantic - weights (ES, Qdrant) {weights}")


    return rrf_fusion([es_docs, qdrant_docs], weights=weights, k=20)
