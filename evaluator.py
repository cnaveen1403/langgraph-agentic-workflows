import ollama


def evaluate_answer(question, context, answer):

    prompt = f"""
You are an evaluator.

Evaluate the answer based on:

1. Relevance to question
2. Use of context
3. Correctness

Give score from 1 to 10.

Question:
{question}

Context:
{context}

Answer:
{answer}

Output ONLY a number.
"""

    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]