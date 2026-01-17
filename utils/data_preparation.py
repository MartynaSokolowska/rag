import datetime
import json
import os

import tqdm
from datasets import load_dataset
from pathlib import Path


def load_data_from_culturax(model, N = 50_000, min_len = 500, max_len = 550, output_file="data/culturax_filtered.jsonl", normalize = True):
    read_data_flag = False
    temp_file = "data/temp"
                      
    ds = load_dataset(
        "uonlp/CulturaX",
        "pl",
        token=True,
        streaming=True
    )

    def fix_date(d):
        if not d:
            return None
        try:
            dt = datetime.strptime(d, "%Y/%m/%d %H:%M:%S")
            return dt.isoformat()  
        except:
            return d

    if read_data_flag:
        stream = ds["train"].iter(batch_size=100)

        os.makedirs("data", exist_ok=True)
        count = 0

        with open(temp_file, "w", encoding="utf-8") as f:
            for batch in stream:
                for i in range(100):
                    text = batch.get("text", "")[i]
                    if text is None:
                        continue

                    if not (min_len <= len(text) <= max_len):
                        continue

                    source = batch.get("source")[i]
                    timestamp = batch.get("timestamp")[i]

                    if source == "" or timestamp == "":
                        continue

                    item = {
                        "id": count,
                        "text": text,
                        "domain": source,
                        "date": fix_date(timestamp),
                    }

                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    count += 1

                if count >= N:
                    break

    def embed_text(text):
        passage = f"passage: {text}" 
        emb = model.encode(passage, normalize_embeddings=normalize)
        return emb.tolist()  

    with open(temp_file, "r", encoding="utf-8") as fin, \
        open(output_file, "w", encoding="utf-8") as fout:
        
        for line in tqdm(fin, desc="Embedding documents"):
            row = json.loads(line)
            text = row.get("text", "")
            if not text:
                continue

            row["vector"] = embed_text(text)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Zapisano embeddingi do {output_file}")

    input_path = Path(temp_file)

    if input_path.exists():
        input_path.unlink()

