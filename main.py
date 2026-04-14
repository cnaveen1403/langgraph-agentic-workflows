from graph.workflow import graph
from evaluator import evaluate_answer

if __name__ == "__main__":

    state = {
        "question": "My name is Naveen",
        "chat_history": []
    }

    # Step 1
    result = graph.invoke(state)
    print("\nQ1 Answer:\n", result["answer"])

    # Step 2 (follow-up)
    state = {
        "question": "What is my name?",
        "chat_history": result["chat_history"]
    }

    result = graph.invoke(state)
    print("\nQ2 Answer:\n", result["answer"])

    # Step 3 (RAG question)
    state = {
        "question": "What is Kubernetes?",
        "chat_history": result["chat_history"]
    }

    result = graph.invoke(state)
    print("\nQ3 Answer:\n", result["answer"])

    score = evaluate_answer(
    question=state["question"],
    context="\n".join(result.get("documents", [])),
    answer=result["answer"])

    print("\nEvaluation Score:", score)