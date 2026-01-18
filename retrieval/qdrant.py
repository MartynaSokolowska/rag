# Run from terminal :)
# !docker run -p 6333:6333 qdrant/qdrant

from datetime import datetime
import json
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams
from qdrant_client.http.models import PointStruct
from sentence_transformers import SentenceTransformer
from qdrant_client.http.models import Filter, FieldCondition, Range, MatchValue


model_name = "intfloat/multilingual-e5-small" 
model = SentenceTransformer(model_name)
client = QdrantClient(host="localhost", port=6333)

def embed_query(q):
    return model.encode(f"query: {q}", normalize_embeddings=True).tolist()

def iso_to_timestamp(date_str):
    dt = datetime.fromisoformat(date_str.replace("Z",""))
    return dt.timestamp() 

def load_data_to_qdrant(data_file):
    collection_name = "culturax_pl"

    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance="Cosine") 
    )

    documents = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            documents.append(doc)

    points = [
        PointStruct(
            id=int(doc["id"]),
            vector=doc["vector"],  
            payload={
                **{k:v for k,v in doc.items() if k != "vector" and k != "date"},
                "date": iso_to_timestamp(doc["date"])
            } 
        )
        for doc in documents
    ]

    batch_size = 500
    for i in range(0, len(points), batch_size):
        client.upsert(
            collection_name="culturax_pl",
            points=points[i:i+batch_size]
        )

def get_knn_qdrant(text, limit=5):
    query_vector = embed_query(text)

    results = client.query_points(
        collection_name="culturax_pl",
        query=query_vector,
        limit = limit
    )

    return [{"id": r.id, "text": r.payload["text"]} for r in results.points]


def query_qdrant_filter_domain(text, domain, top_k=3):
    vector = embed_query(text)
    query_filter = Filter(
        must=[
            FieldCondition(key="domain", match=MatchValue(value = domain)),
        ]
    )
    results = client.query_points(
        collection_name="culturax_pl",
        query=vector,
        limit=top_k,
        query_filter=query_filter
    )
    return results

def query_qdrant_filter_date(text, date_from="2000-01-01T00:00:00Z", date_to="2020-01-01T00:00:00Z", top_k=3):
    vector = embed_query(text)
    query_filter = Filter(
        must=[
            FieldCondition(key="date", range=Range(gte=iso_to_timestamp(date_from), lte=iso_to_timestamp(date_to)))
        ]
    )
    results = client.query_points(
        collection_name="culturax_pl",
        query=vector,
        limit=top_k,
        query_filter=query_filter
    )
    return results

