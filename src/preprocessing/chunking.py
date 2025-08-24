import pandas as pd
from src.config import DATA_DIR 
def chunk_documents(docs, chunk_size=1):
    chunks = []
    for i, doc in enumerate(docs):
        paragraphs = [p.strip() for p in doc.split("\n") if p.strip()]
        for para in paragraphs:
            chunks.append({
                "paper_id": i,
                "chunk_text": para
            })
    df = pd.DataFrame(chunks)
    df.to_csv(f"{DATA_DIR}/chunks.csv", index=False)
    return df