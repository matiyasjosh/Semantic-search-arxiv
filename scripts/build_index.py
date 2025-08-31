import pandas as pd
from src.ingestion.arxiv_scrapper import fetch_many
from src.ingestion.parser import dedupe_metadata, save_metadata
from src.preprocessing.chunking import chunk_documents_from_metadata
from src.embeddings.embed import compute_tfidf, compute_transformer_embeddings, build_and_save_faiss
from src.config import DATA_DIR

# 1) fetch many papers (1000+)
metadata_df = fetch_many(query="cat:cs.AI", total_results=1200, batch_size=100, sleep_between=0.8)
print("Fetched", len(metadata_df), "rows")

# 2) clean & dedupe
metadata_df = dedupe_metadata(metadata_df)
save_metadata(metadata_df)

# 3) chunk
chunks_df = chunk_documents_from_metadata(metadata_df)

# 4) TF-IDF artifacts
vectorizer, tfidf_matrix = compute_tfidf(chunks_df)

# 5) transformer embeddings + faiss
model, embeddings = compute_transformer_embeddings(chunks_df)
index = build_and_save_faiss(embeddings)
