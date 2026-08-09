from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# 1. Load vector store created by create_database.py
embedding_model = MistralAIEmbeddings()
vectorstore = Chroma(
    persist_directory="chroma-db",
    embedding_function=embedding_model
)

# 2. Set up retriever
retriever = vectorstore.as_retriever(search_type = "mmr", search_kwargs={"k": 3, "fetch_k": 10 , "lambda_mult": 0.5})

#llms
llm = ChatMistralAI(model="mistral-small-2506")

# 3. Define prompt template with context
template = ChatPromptTemplate.from_messages([
    ("system", "You are an AI instructor that summarizes and answers questions based on context:\n\n{context} , if answer doesnto exist tell not found"),
    ("human", """Context: {context}
    
    Question: {question}
    """)
])


print("RAG system created")


print("press 0 to exit")


while True:
    query = input("You ->")
    if query == "0":
        print("Exiting")
        break
    docs = retriever.invoke(query)


    context =  ".\n\n".join([doc.page_content for doc in docs])
    final_prompt = template.invoke({"context":context,"question":query})
    
    response = llm.invoke(final_prompt)
    print("Answer")
    print(response.content)


    
#model = ChatMistralAI(model="mistral-small-latest")

# # 4. Query vector store and invoke model
# query = "What are the main data structures covered in this document?"


# docs = retriever.invoke(query)
# context = "\n\n".join([doc.page_content for doc in docs])

# chain = template | model
# response = chain.invoke({"context": context, "question": query})

# print(f"Question: {query}\n")
# print("Answer:")
# print(response.content)
