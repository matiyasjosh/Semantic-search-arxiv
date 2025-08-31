import os

DATA_DIR = "data"
DOWNLOAD_DIR = "downloads"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 100
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
USER_AGENT = "arXivScraper/0.1 (https://github.com/matijosh/Semantic-search-arxiv)"