import ollama
from typing import TypedDict, List
import ollama
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    question: str
    documents: List[str]
    decision: str
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

def agent_reason_node(state: AgentState):
    question = state["question"]

    prompt = f"""
Decide if the question needs document retrieval.

Question:
{question}

Answer with only one word:
YES or NO
"""
    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}]
    )

    decision = response["message"] ["content"].strip().upper()

    return {"decision": decision}

def tool_node(state: AgentState):

    question = state["question"]

    # simple simulated RAG
    docs = [
        "FastAPI is a modern Python framework used to build APIs.",
        "It supports asynchronous programming and high performance."
    ]

    return {"documents": docs}

def generate_answer_node(state: AgentState):

    question = state["question"]
    docs = state.get("documents", [])

    context = "\n".join(docs)

    prompt = f"""
Use the context to answer the question.

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

def decide_tool(state: AgentState):

    if state["decision"] == "YES":
        return "tool"

    return "generate"

builder = StateGraph(AgentState)

builder.add_node("agent_reason", agent_reason_node)
builder.add_node("tool", tool_node)
builder.add_node("generate", generate_answer_node)

builder.add_edge(START, "agent_reason")

builder.add_conditional_edges(
    "agent_reason",
    decide_tool,
    {
        "tool": "tool",
        "generate": "generate"
    }
)

builder.add_edge("tool", "generate")
builder.add_edge("generate", END)

graph = builder.compile()