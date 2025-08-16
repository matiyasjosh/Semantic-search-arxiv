def chunk_documents(docs, chunk_size=1):
    chunks = []
    for doc in docs:
        paragraphs = [p.strip() for p in doc.split("\n") if p.strip()]
        chunks.extend(paragraphs)
    return chunks