import arxiv
import pandas as pd
from src.config import DATA_DIR

def fetch_arxiv(query="cat:cs.AI", max_results=100, sort_by=arxiv.SortCriterion.SubmittedDate):
    search = arxiv.Search(query=query, max_results=max_results, sort_by=sort_by)
    client = arxiv.Client()

    results = []
    for result in client.results(search):
        results.append({
            "Title": result.title,
            "Authors": ", ".join([author.name for author in result.authors]),
            "Date": result.published.date(),
            "Summary": result.summary.strip().replace("\n", " "),
            "pdf_url": result.pdf_url
        })

    df = pd.DataFrame(results)
    print(df)
    df.to_csv(f"{DATA_DIR}/metadata.csv", index=False)
    return df