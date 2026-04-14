from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
import ollama

from graph.rag_tool import search_documents


# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def is_personal_question(question):
    keywords = ["my name", "who am i"]
    return any(k in question.lower() for k in keywords)


def is_relevant(question, docs):
    q_words = set(question.lower().split())
    doc_words = set(" ".join(docs).lower().split())
    overlap = q_words.intersection(doc_words)
    return len(overlap) > 1


# ---------------------------
# STATE DEFINITION
# ---------------------------
class AgentState(TypedDict):
    question: str
    documents: List[str]
    answer: str
    decision: str
    chat_history: List[str]


# ---------------------------
# SUPERVISOR NODE
# ---------------------------
def supervisor_node(state: AgentState):

    question = state["question"]

    prompt = f"""
You are a strict decision system.

Rules:
- If the question needs factual or external knowledge → RESEARCH
- If the question is personal or conversational → WRITE

Examples:
- "What is FastAPI?" → RESEARCH
- "What is my name?" → WRITE
- "Explain RAG" → RESEARCH

Question:
{question}

Answer ONLY one word:
RESEARCH or WRITE
"""

    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}]
    )

    decision = response["message"]["content"].strip().upper()

    if "RESEARCH" in decision:
        decision = "RESEARCH"
    else:
        decision = "WRITE"

    print("\n---- SUPERVISOR ----")
    print("Question:", question)
    print("Decision:", decision)

    return {"decision": decision}


# ---------------------------
# RESEARCH NODE
# ---------------------------
def research_agent(state: AgentState):

    question = state["question"]
    docs = search_documents(question)

    print("\n---- RESEARCH ----")
    print("Query:", question)
    print("Docs:", docs)

    return {"documents": docs}


# ---------------------------
# WRITER NODE
# ---------------------------
def writer_agent(state: AgentState):

    question = state["question"]
    docs = state.get("documents", [])
    chat_history = state.get("chat_history", [])

    # ---------------------------
    # 🧠 GUARDRAILS (BEFORE LLM)
    # ---------------------------

    if is_personal_question(question):
        pass

    elif not docs:
        return {
            "answer": "I don’t have enough information to answer this.",
            "chat_history": chat_history
        }

    elif not is_relevant(question, docs):
        return {
            "answer": "The retrieved information does not seem relevant.",
            "chat_history": chat_history
        }

    # ---------------------------
    # GENERATION
    # ---------------------------

    context = "\n".join(docs)
    history_text = "\n".join(chat_history)

    prompt = f"""
You are a helpful assistant.

Conversation history:
{history_text}

Context:
{context}

Question:
{question}

Answer clearly and concisely.
"""

    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response["message"]["content"]

    print("\n---- WRITER ----")
    print("Context:", context)
    print("Answer:", answer)

    # ---------------------------
    # MEMORY UPDATE
    # ---------------------------

    new_history = chat_history.copy()
    new_history.append(f"User: {question}")
    new_history.append(f"Assistant: {answer}")

    return {
        "answer": answer,
        "chat_history": new_history
    }


# ---------------------------
# ROUTER
# ---------------------------
def supervisor_router(state: AgentState):

    if state["decision"] == "RESEARCH":
        return "research"

    return "writer"


# ---------------------------
# GRAPH BUILDING
# ---------------------------
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