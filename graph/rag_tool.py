import faiss
import numpy as np
import ollama


# ----- LOAD DOCUMENTS -----
with open("documents.txt") as f:
    texts = f.read().split("\n")


# ----- EMBEDDING FUNCTION -----
def get_embedding(text):
    response = ollama.embeddings(
        model="nomic-embed-text",
        prompt=text
    )
    return np.array(response["embedding"], dtype="float32")


# ----- BUILD INDEX -----
embeddings = [get_embedding(t) for t in texts if t.strip()]
dim = len(embeddings[0])

index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings))


# ----- SEARCH FUNCTION -----
def search_documents(query, k=2):

    query_vec = get_embedding(query)

    D, I = index.search(np.array([query_vec]), k)

    results = [texts[i] for i in I[0]]

    return results