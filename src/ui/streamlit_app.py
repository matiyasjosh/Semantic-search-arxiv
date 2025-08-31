import faiss
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from src.config import MODEL_NAME, DATA_DIR
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------
# Load model, FAISS index, and metadata
# --------------------------
st.set_page_config(page_title="Semantic arXiv Search", page_icon="📚", layout="wide")

st.title("📚 Semantic arXiv Search")

@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

@st.cache_resource
def load_index():
    return faiss.read_index(f"{DATA_DIR}/arxiv_index.faiss")

@st.cache_data
def load_metadata():
    return pd.read_csv(f"{DATA_DIR}/metadata.csv")


model = load_model()
index = load_index()
metadata = load_metadata()
vectorizer = joblib.load(f"{DATA_DIR}/vectorizer.pkl")
tfidf_matrix = joblib.load(f"{DATA_DIR}/tfidf_matrix.pkl")



def hybridSearch(query, model, faiss_index, k=5, alpha=0.6):
    # Embedding query
    query_vec = model.encode([query], convert_to_numpy=True, show_progress_bar=False)
    query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)
    faiss_score, faiss_indices = faiss_index.search(query_vec, k*3)

    # TFIDF score
    tfidf_vec = vectorizer.transform([query])
    tfidf_score = cosine_similarity(tfidf_vec, tfidf_matrix).flatten()

    # Combined score
    result = []
    for i, idx in enumerate(faiss_indices[0]):
        combined_score = alpha * faiss_score[0][i] + (1 - alpha) * tfidf_score[idx]
        result.append((combined_score, idx))
    
    result = sorted(result, key=lambda x: x[1], reverse=True)
    return result


# --------------------------
# Query input
# --------------------------
query = st.text_input("🔍 Enter your query:")

if query:
    chunks = pd.read_csv(f"{DATA_DIR}/chunks.csv")   # stores chunk_text + paper_id
    result = hybridSearch(query, model, index, chunks)
   
    for rank, (idx, score) in enumerate(result, 1):
        snippet = chunks.iloc[int(idx)]["chunk_text"]
        paper_id = chunks.iloc[int(idx)]["paper_id"]
        paper = metadata.iloc[paper_id]

        st.markdown(f"### {rank}. {paper['Title']}")
        st.write(f"**Similarity:** {score:.4f}")
        st.write(f"**Snippet:** {snippet[:300]}...")
        st.write(f"👨‍🔬 {paper['Authors']} | 📅 {paper['Date']}")
        st.markdown(f"[📄 Read PDF]({paper['pdf_url']})")
        st.divider()


# when running use `PYTHONPATH=. streamlit run src/ui/streamlit_app.py` while being on the parent folder