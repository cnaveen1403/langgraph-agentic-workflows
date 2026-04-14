import os
import numpy as np
import ollama


# ---------------------------
# LOAD DOCUMENTS
# ---------------------------
BASE_DIR = os.path.dirname(__file__)
file_path = os.path.join(BASE_DIR, "..", "documents.txt")

with open(file_path) as f:
    texts = [t.strip() for t in f.readlines() if t.strip()]


# ---------------------------
# EMBEDDING FUNCTION
# ---------------------------
def get_embedding(text):
    response = ollama.embeddings(
        model="nomic-embed-text",
        prompt=text
    )
    return np.array(response["embedding"], dtype="float32")


# ---------------------------
# BUILD EMBEDDINGS
# ---------------------------
doc_embeddings = [get_embedding(t) for t in texts]


# ---------------------------
# COSINE SIMILARITY
# ---------------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ---------------------------
# SIMPLE RERANKER
# ---------------------------
def simple_rerank(query, docs):
    """
    Boost definition-like sentences (heuristic)
    """
    return sorted(
        docs,
        key=lambda x: (" is " in x.lower(), len(x)),
        reverse=True
    )


# ---------------------------
# SEARCH FUNCTION (CORE RAG)
# ---------------------------
def search_documents(query, k=5):

    query_vec = get_embedding(query)

    # similarity calculation
    similarities = [
        cosine_similarity(query_vec, doc_vec)
        for doc_vec in doc_embeddings
    ]

    # get top-k indices
    top_k_indices = np.argsort(similarities)[-k:][::-1]

    # retrieve documents
    initial_docs = [texts[i] for i in top_k_indices]

    # remove duplicates (safety)
    seen = set()
    unique_docs = []
    for doc in initial_docs:
        if doc not in seen:
            unique_docs.append(doc)
            seen.add(doc)

    # rerank
    reranked_docs = simple_rerank(query, unique_docs)

    # return top 2
    return reranked_docs[:2]