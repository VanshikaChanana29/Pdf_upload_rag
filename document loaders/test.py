from langchain_community.document_loaders import TextLoader

from langchain_text_splitters import CharacterTextSplitter


text_splitter = CharacterTextSplitter(
    chunk_size=10,
    chunk_overlap=1
)

loader = TextLoader("document loaders/notes.txt")

docs = loader.load()
chunks = text_splitter.split_documents(docs)

print(chunks)