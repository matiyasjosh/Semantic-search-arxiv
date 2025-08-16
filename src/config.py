import os

DATA_DIR = "data"
DOWNLOAD_DIR = "downloads"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
