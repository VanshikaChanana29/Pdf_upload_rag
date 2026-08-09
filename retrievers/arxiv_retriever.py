from langchain_community.retrievers import ArxivRetriever


# Create the retriever
retriever = ArxivRetriever(
    load_max_docs=2,
    load_all_available_meta=True
)


# Query arXiv
docs = retriever.invoke("large language models")


# Print results
for i, doc in enumerate(docs):
    print(f"\nResult {i + 1}")
    print("=" * 80)

    print("Title:", doc.metadata.get("Title", "N/A"))
    print("Authors:", doc.metadata.get("Authors", "N/A"))
    print("Published:", doc.metadata.get("Published", "N/A"))
    print("URL:", doc.metadata.get("Entry ID", "N/A"))

    print("\nContent:")
    print(doc.page_content[:1000])

    print("=" * 80)