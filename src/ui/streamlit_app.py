import streamlit as st
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from src.config import MODEL_NAME, DATA_DIR

st.set_page_config(page_title="Semantic Search", page_icon=":mag_right:", layout="wide")

st.title("Semantic Arxiv Search")

@st.cache_resource
def load_model():
    model = SentenceTransformer(MODEL_NAME)
    return model

import streamlit as st
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from src.config import MODEL_NAME, DATA_DIR

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

# --------------------------
# Query input
# --------------------------
query = st.text_input("🔍 Enter your query:")

if query:
    # Encode query
    query_vec = model.encode([query])
    query_vec = np.array(query_vec).astype("float32")

    # Search top-k
    k = st.slider("Top K results", 1, 20, 5)
    distances, indices = index.search(query_vec, k)

    # --------------------------
    # Display results
    # --------------------------
    for i, idx in enumerate(indices[0]):
        paper = metadata.iloc[idx]
        st.markdown(f"### {i+1}. {paper['Title']}")
        st.write(f"**Similarity:** {distances[0][i]:.4f}")
        st.write(paper['Summary'])  # we stored abstract as "Summary"
        st.markdown(f"[📄 Read PDF]({paper['pdf_url']})")
        st.divider()
