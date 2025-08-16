import os, requests
from src.config import DOWNLOAD_DIR

def download_pdf(metadata):
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    for _, row in metadata.iterrows():
        title = row["Title"].replace(" ", "_").replace("/", "_")
        pdf_url = row["pdf_url"]
        response = requests.get(pdf_url)

        if response.status_code == 200:
            with open(f"{DOWNLOAD_DIR}/{title}.pdf", "wb") as f:
                f.write(response.content)
        else:
            print(f"Failed to download {title}")