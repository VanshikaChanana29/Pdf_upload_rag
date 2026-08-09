import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings

load_dotenv()

db = Chroma(
    persist_directory="chroma-db",
    embedding_function=MistralAIEmbeddings()
)

data = db._collection.get()

print("=" * 60)
print("CHROMA DB COLLECTION SUMMARY")
print(f"Total Chunks Stored: {len(data['ids'])}")
print("=" * 60)

# Group chunks by file source
breakdown = {}
for meta in data["metadatas"]:
    src = os.path.basename(meta.get("source", "Unknown"))
    breakdown[src] = breakdown.get(src, 0) + 1

print("\nDocuments Breakdown in ChromaDB:")
for doc, count in breakdown.items():
    print(f"  - {doc}: {count} chunk(s)")

print("\n" + "=" * 60)
print("SAMPLE CHUNK FROM REACT INTERVIEW PDF:")
print("=" * 60)

for idx, meta in enumerate(data["metadatas"]):
    if "React" in meta.get("source", ""):
        print(f"Chunk ID: {data['ids'][idx]}")
        print(f"Source File: {meta.get('source')}")
        print(f"Page Number: {meta.get('page', 'N/A')}")
        print(f"Content Snippet:\n{data['documents'][idx][:300]}...\n")
        break
