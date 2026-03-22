# LangGraph Agentic Workflows

This project demonstrates a production-style AI agent system using:

- LangGraph for orchestration
- Multi-agent architecture (Supervisor pattern)
- RAG (Retrieval Augmented Generation)
- FAISS vector database
- Local LLM using Ollama

## Architecture

User Query
    ↓
Supervisor Agent
    ↓
Research Agent (FAISS retrieval)
    ↓
Writer Agent (LLM response)
    ↓
Final Answer

## Features

- Agent orchestration using LangGraph
- Tool-based RAG integration
- Multi-agent workflow design
- Vector similarity search

## Run

```bash
python main.py