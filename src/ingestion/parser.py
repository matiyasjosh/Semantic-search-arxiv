import pandas as pd
import re
from src.config import DATA_DIR

def normalize_title(t):
    t = re.sub(r'\s+', ' ', (t or "").lower()).strip()
    # remove punctuation for simpler matching
    return re.sub(r'[^\w\s]', '', t)

def dedupe_metadata(df: pd.DataFrame):
    # prefer arXiv id for dedupe
    if "arxiv_id" in df.columns:
        df = df.dropna(subset=["arxiv_id"]).drop_duplicates(subset=["arxiv_id"])
    # still handle duplicates by normalized title
    df["norm_title"] = df["title"].apply(normalize_title)
    df = df.drop_duplicates(subset=["norm_title"])
    df = df.reset_index(drop=True)
    df = df.drop(columns=["norm_title"])
    return df

def save_metadata(df: pd.DataFrame, path=None):
    path = path or f"{DATA_DIR}/metadata.csv"
    df.to_csv(path, index=False)
    print(f"Saved metadata to {path}")
