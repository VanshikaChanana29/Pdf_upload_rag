import os
import shutil
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings

load_dotenv()

docs_dir = "document loaders"
print(f"Scanning '{docs_dir}' for documents...")

docs = []
loaded_files = []

if os.path.exists(docs_dir):
    for filename in os.listdir(docs_dir):
        filepath = os.path.join(docs_dir, filename)
        if filename.endswith(".pdf"):
            try:
                pdf_docs = PyPDFLoader(filepath).load()
                if pdf_docs:
                    docs.extend(pdf_docs)
                    loaded_files.append(filename)
                    print(f"Loaded PDF: {filename} ({len(pdf_docs)} page(s))")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        elif filename.endswith(".txt"):
            try:
                txt_docs = TextLoader(filepath, encoding="utf-8").load()
                if txt_docs:
                    docs.extend(txt_docs)
                    loaded_files.append(filename)
                    print(f"Loaded TXT: {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")

print(f"\nTotal loaded: {len(docs)} document page(s)/file(s) from files: {loaded_files}")

if docs:
    # Remove stale chroma-db directory to prevent old data pollution
    if os.path.exists("chroma-db"):
        print("Resetting old 'chroma-db' directory...")
        shutil.rmtree("chroma-db", ignore_errors=True)

    print("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} text chunks.")

    print("Generating embeddings and saving to Chroma database...")
    embedding_model = MistralAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="chroma-db"
    )
    print("Database created successfully in 'chroma-db'!")
else:
    print("No valid documents found in 'document loaders/'.")