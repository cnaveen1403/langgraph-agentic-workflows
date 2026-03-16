from graph.workflow import graph

if __name__ == "__main__":

    result = graph.invoke({
        "question": "What is FastAPI?"
    })

    print("\nAnswer:\n")
    print(result["answer"])