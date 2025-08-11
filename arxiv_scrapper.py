import arxiv
import pandas as pd

# Define your query and settings
search = arxiv.Search(
    query="cat:cs.AI",  # Change to cs.CV or any other category
    max_results=5,
    sort_by=arxiv.SortCriterion.SubmittedDate
)

client = arxiv.Client()

# Collect metadata
results = []
for result in client.results(search):
    results.append({
        "Title": result.title,
        "Authors": ", ".join([author.name for author in result.authors]),
        "Date": result.published.date(),
        "Summary": result.summary.strip().replace("\n", " "),
        "pdf_url": result.pdf_url
    })

    print

# Display as a table
df = pd.DataFrame(results)
df.to_csv("data.csv", index=False)

print(df)
