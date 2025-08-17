import faiss
import numpy as np
from src.ingestion.arxiv_scrapper import fetch_arxiv
from src.preprocessing.chunking import chunk_documents
from src.embeddings.transformer_embedder import TransformerEmbedder
from src.config import MODEL_NAME, DATA_DIR

# fetch metadata
df = fetch_arxiv("cat:cs.AI", max_results=200)

# create chunks (from abstracts/summaries)
chunks = chunk_documents(df["Summary"].tolist())

# generate embeddings
embedder = TransformerEmbedder(MODEL_NAME)
embeddings = embedder.encode(chunks, to_tensor=False)

# Normalize embeddings
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# build FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

# save FAISS index
index_path = f"{DATA_DIR}/arxiv_index.faiss"
faiss.write_index(index, index_path)
print(f"Saved FAISS index to {index_path}")

print("Build complete! Metadata and FAISS index are ready for Streamlit.")
