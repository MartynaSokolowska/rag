def chunk_by_chars(text, chunk_size=550):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end

    return chunks

def chunk_by_tokens(text, tokens=200):
    words = text.split() 
    chunks = []
    for i in range(0, len(words), tokens):
        chunk = " ".join(words[i:i+tokens])
        chunks.append(chunk)
    return chunks