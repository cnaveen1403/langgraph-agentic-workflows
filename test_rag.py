from graph.rag_tool import search_documents

query = "What is FastAPI?"

results = search_documents(query)

print("\nQuery:", query)
print("\nRetrieved Documents:")

for i, doc in enumerate(results, 1):
    print(f"{i}. {doc}")