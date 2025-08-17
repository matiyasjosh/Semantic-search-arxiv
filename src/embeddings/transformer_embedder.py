from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

class TransformerEmbedder:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts, to_tensor=True):
        return self.model.encode(texts, convert_to_tensor=to_tensor, show_progress_bar=True)

    def semantic_search(self, query, corpus_embeddings, chunks, top_k=5):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        similarities = cos_sim(query_embedding, corpus_embeddings)[0]
        top_indices = similarities.argsort(descending=True)[:top_k]
        return [(chunks[i], float(similarities[i])) for i in top_indices]
