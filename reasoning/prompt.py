import ollama
from config.config import LLM_MODEL, TASK_PROMPT


def build_context_prompt(user_input, chunks, prompt=TASK_PROMPT):
    context = "\n".join([f"[{i}] {chunk}" for i, chunk in enumerate(chunks, start=1)])

    return f"""
    {prompt}

    KONTEKST:
    {context}

    PYTANIE:
    {user_input}
    """


def call_llm(content, system=None, shots=None):
    messages = []

    if system is not None:
        messages.append({
            "role": "system",
            "content": system
        })

    if shots is not None:
        messages.extend(shots)

    messages.append({
        "role": "user",
        "content": content
    })

    response = ollama.chat(
        model=LLM_MODEL,
        messages=messages,
    )

    return response["message"]["content"]
