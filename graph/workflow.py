from typing import TypedDict, List
import ollama
from langgraph.graph import StateGraph, START, END


# -------------------
# STATE
# -------------------
class AgentState(TypedDict):
    question: str
    documents: List[str]
    answer: str
    decision: str


# -------------------
# SUPERVISOR
# -------------------
def supervisor_node(state: AgentState):

    question = state["question"]

    prompt = f"""
You are a supervisor.

Decide the next step:

Options:
- RESEARCH (if external info needed)
- WRITE (if you can answer directly)

Question:
{question}

Answer with one word: RESEARCH or WRITE
"""

    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}]
    )

    decision = response["message"]["content"].strip().upper()

    return {"decision": decision}


# -------------------
# RESEARCH AGENT
# -------------------
def research_agent(state: AgentState):

    docs = [
        "FastAPI is a modern Python framework used to build APIs.",
        "It supports async programming and high performance."
    ]

    return {"documents": docs}


# -------------------
# WRITER AGENT
# -------------------
def writer_agent(state: AgentState):

    question = state["question"]
    docs = state.get("documents", [])

    context = "\n".join(docs)

    prompt = f"""
Write a clear answer.

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


# -------------------
# ROUTER
# -------------------
def supervisor_router(state: AgentState):

    if state["decision"] == "RESEARCH":
        return "research"

    return "writer"


# -------------------
# GRAPH
# -------------------
builder = StateGraph(AgentState)

builder.add_node("supervisor", supervisor_node)
builder.add_node("research", research_agent)
builder.add_node("writer", writer_agent)

builder.add_edge(START, "supervisor")

builder.add_conditional_edges(
    "supervisor",
    supervisor_router,
    {
        "research": "research",
        "writer": "writer"
    }
)

builder.add_edge("research", "writer")
builder.add_edge("writer", END)

graph = builder.compile()