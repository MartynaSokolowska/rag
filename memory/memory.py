import json
import os
from datetime import datetime

MEMORY_FILE = "pending.json"


def _init_memory():
    if not os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump({"pending_queries": []}, f, ensure_ascii=False, indent=2)


def add_to_memory(query):
    _init_memory()

    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        memory = json.load(f)

    next_id = (
        max((q["id"] for q in memory["pending_queries"]), default=0) + 1
    )

    entry = {
        "id": next_id,
        "query": query,
        "status": "pending",
        "timestamp": datetime.utcnow().isoformat()
    }

    memory["pending_queries"].append(entry)

    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)


def read_memory(status="pending"):
    _init_memory()

    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        memory = json.load(f)

    return [
        q for q in memory["pending_queries"]
        if q.get("status") == status
    ]

def is_query_in_memory(query: str) -> bool:
    _init_memory()

    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        memory = json.load(f)

    return any(
        q["query"].strip().lower() == query.strip().lower()
        for q in memory["pending_queries"]
    )
