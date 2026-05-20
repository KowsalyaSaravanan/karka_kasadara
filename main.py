"""
main.py
-------
Entry point. Runs 3 sample queries and prints results.
Also supports interactive mode.
"""

from rag import RAGSystem


SAMPLE_QUERIES = [
    "What is the admission process at Velammal schools?",
    "How are fee payments structured and what happens if fees are delayed?",
    "What extracurricular activities does Velammal offer?",
]


def print_result(result: dict):
    print("=" * 70)
    print(f"Query: {result['query']}")
    print("-" * 70)
    for chunk_info in result["retrieved_chunks"]:
        print(f"\n[Rank {chunk_info['rank']} | Score: {chunk_info['score']}]")
        print(chunk_info["chunk"])
    print("-" * 70)
    print(f"Answer: {result['answer']}")
    print("=" * 70)
    print()


def main():
    rag = RAGSystem()

    print("\n### Running 3 Sample Queries ###\n")
    for q in SAMPLE_QUERIES:
        result = rag.query(q)
        print_result(result)

    # Interactive mode
    print("\nEnter your own query (or 'exit' to quit):")
    while True:
        user_input = input(">> ").strip()
        if user_input.lower() in ("exit", "quit", ""):
            break
        result = rag.query(user_input)
        print_result(result)


if __name__ == "__main__":
    main()
