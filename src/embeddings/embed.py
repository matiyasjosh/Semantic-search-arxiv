from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import numpy as np
import faiss
from src.config import DATA_DIR, MODEL_NAME

def compute_tfidf(chunks_df):
    vectorizer = TfidfVectorizer(max_features=50000)
    tfidf_matrix = vectorizer.fit_transform(chunks_df["chunk_text"].tolist())
    joblib.dump(vectorizer, f"{DATA_DIR}/vectorizer.pkl")
    joblib.dump(tfidf_matrix, f"{DATA_DIR}/tfidf_matrix.pkl")
    print("Saved TF-IDF artifacts")
    return vectorizer, tfidf_matrix

def compute_transformer_embeddings(chunks_df):
    model = SentenceTransformer(MODEL_NAME)
    texts = chunks_df["chunk_text"].tolist()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    # normalize
    embeddings = embeddings.astype("float32")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms==0] = 1.0
    embeddings = embeddings / norms
    return model, embeddings

def build_and_save_faiss(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, f"{DATA_DIR}/arxiv_index.faiss")
    print(f"Saved FAISS index ({embeddings.shape[0]} vectors) to {DATA_DIR}/arxiv_index.faiss")
    return index
