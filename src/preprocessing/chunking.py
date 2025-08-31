import pandas as pd
from src.config import DATA_DIR

def chunk_documents_from_metadata(metadata_df, min_paragraph_chars=40):
    rows = []
    for paper_id, row in metadata_df.iterrows():
        text = row.get("summary", "") or ""
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        if not paragraphs:
            # fallback: split on sentences or whitespace
            paragraphs = [t for t in text.split(". ") if len(t.strip()) > min_paragraph_chars]
        for para in paragraphs:
            if len(para) < min_paragraph_chars:
                continue
            rows.append({"paper_id": int(paper_id), "chunk_text": para})
    chunks_df = pd.DataFrame(rows)
    chunks_df.to_csv(f"{DATA_DIR}/chunks.csv", index=False)
    print(f"Saved {len(chunks_df)} chunks to {DATA_DIR}/chunks.csv")
    return chunks_df
