from typing import TypedDict, List
import ollama
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    question: str
    documents: List[str]
    answer: str

def retrieve_node(state: AgentState):
    question = state["question"]

    docs = [
        "FastApi is a modern Python framework used to build APIs.",
        "It supports asynchronous programming and high performance."
    ]

    return {"documents": docs}


def generate_node(state: AgentState):

    question = state["question"]
    docs = state["documents"]

    context = "\n".join(docs)

    prompt = f"""
Use the context below to answer the question.

Context:
{context}

Question:
{question}
"""

    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}]
    )

    return {"answer": response["message"]["content"]}

builder = StateGraph(AgentState)

builder.add_node("retrieve", retrieve_node)
builder.add_node("generate", generate_node)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)

graph = builder.compile()