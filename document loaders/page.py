from langchain_community.document_loaders import WebBaseLoader
url  = "https://www.apple.com/in/shop/buy-mac/macbook-air"

data = WebBaseLoader(url)
docs = data.load()

print(docs)
