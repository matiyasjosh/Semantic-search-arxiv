import arxiv
import time
import pandas as pd
from src.config import DATA_DIR, USER_AGENT, BATCH_SIZE
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

@retry(wait=wait_exponential(multiplier=1, min=2, max=30), stop=stop_after_attempt(5), retry = retry_if_exception_type(Exception))
def fetch_batch(query, start, max_results):
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        start=start,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    client = arxiv.Client(user_agent=USER_AGENT, page_size=max_results)
    return list(client.results(search))

def fetch_many(query="cat:cs.AI", total_results=1000, batch_size=BATCH_SIZE, sleep_between=1.0):
    all_entries = []
    for start in range(0, total_results, batch_size):
        try:
            batch = fetch_batch(query, start, batch_size)
        except Exception as e:
            print(f"Failed to fetch batch start={start}: {e}")
            break
        if not batch:
            print(f"No more results at start={start}. Ending.")
            break
        all_entries.extend(batch)
        time.sleep(sleep_between)  # be polite and avoid rate limits
    # convert to DataFrame
    rows = []
    for r in all_entries:
        rows.append({
            "arxiv_id": getattr(r, "entry_id", None) and r.entry_id.rsplit("/", 1)[-1],
            "title": r.title.strip() if r.title else "",
            "authors": ", ".join([a.name for a in r.authors]) if r.authors else "",
            "date": r.published.date() if r.published else None,
            "summary": (r.summary or "").strip().replace("\n", " "),
            "pdf_url": r.pdf_url or "",
            "entry_id": r.entry_id or ""
        })
    
    return pd.DataFrame(rows)