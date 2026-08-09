import os
import streamlit as st
from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter



# ============================================================
# CONFIG & ENVIRONMENT
# ============================================================

load_dotenv()

# Resolve Mistral API Key (supporting Streamlit Secrets in cloud and .env locally)
mistral_api_key = None
try:
    if "MISTRAL_API_KEY" in st.secrets:
        mistral_api_key = str(st.secrets["MISTRAL_API_KEY"]).strip().strip('"').strip("'")
except Exception:
    pass

if not mistral_api_key:
    mistral_api_key = (os.getenv("MISTRAL_API_KEY") or "").strip().strip('"').strip("'")

if mistral_api_key:
    os.environ["MISTRAL_API_KEY"] = mistral_api_key

st.set_page_config(
    page_title="RAG AI Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM STYLING (CSS)
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at 10% 10%, rgba(99, 102, 241, 0.12), transparent 30%),
                    radial-gradient(circle at 90% 20%, rgba(168, 85, 247, 0.10), transparent 30%),
                    #0b0f19;
        color: #f8fafc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 5rem;
        max-width: 1200px;
    }

    section[data-testid="stSidebar"] {
        background: #090d16;
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    .hero {
        padding: 35px 40px;
        border-radius: 24px;
        margin-bottom: 25px;
        background: linear-gradient(135deg, rgba(99,102,241,0.18), rgba(168,85,247,0.10));
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 20px 60px rgba(0,0,0,0.25);
    }

    .hero-title {
        font-size: 40px;
        font-weight: 800;
        margin-bottom: 8px;
        background: linear-gradient(90deg, #ffffff, #a5b4fc, #c4b5fd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 16px;
        line-height: 1.6;
    }

    .status-card {
        background: rgba(15,23,42,0.75);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px;
        padding: 18px;
        text-align: center;
    }

    .status-title {
        color: #94a3b8;
        font-size: 13px;
    }

    .status-value {
        color: #f8fafc;
        font-size: 18px;
        font-weight: 700;
        margin-top: 5px;
    }

    .online {
        color: #4ade80;
    }

    .source-card {
        background: rgba(15,23,42,0.75);
        border: 1px solid rgba(168,85,247,0.2);
        border-radius: 10px;
        padding: 10px 14px;
        margin-top: 8px;
        margin-bottom: 8px;
    }

    .source-title {
        color: #c4b5fd;
        font-weight: 600;
        font-size: 13px;
    }

    .source-text {
        color: #94a3b8;
        font-size: 12px;
        margin-top: 2px;
    }

    .welcome-card {
        background: rgba(15,23,42,0.6);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 18px;
        padding: 20px;
        height: 140px;
    }

    .welcome-icon {
        font-size: 24px;
        margin-bottom: 6px;
    }

    .welcome-title {
        font-weight: 700;
        color: #f8fafc;
    }

    .welcome-text {
        font-size: 13px;
        color: #94a3b8;
        margin-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# RAG PIPELINE LOADING
# ============================================================

class CustomDoc:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata



@st.cache_resource
def load_rag():
    current_api_key = os.getenv("MISTRAL_API_KEY", "").strip()
    if not current_api_key:
        raise ValueError("MISTRAL_API_KEY is missing! Please configure MISTRAL_API_KEY in Streamlit Secrets or environment variables.")

    embedding_model = MistralAIEmbeddings(api_key=current_api_key)

    vectorstore = Chroma(
        persist_directory="chroma-db",
        embedding_function=embedding_model
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    llm = ChatMistralAI(api_key=current_api_key, model="mistral-small-2506")

    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an AI instructor and RAG document assistant.

Answer the user's question accurately using the provided context.

If the user asks about a specific document or resume (such as 'Vanshika_Chanana_SDE_Resume (2).pdf', 'dsa.pdf', 'resume', etc.) or asks an overview question ("what is this?", "what do you know?", "summarize"), provide a clear summary of the relevant details and topics found in the provided context.

If the question is completely unrelated to the provided context, say:
"I couldn't find this information in the provided documents."

Context:
{context}"""
        ),
        (
            "human",
            """Question:
{question}"""
        )
    ])

    return vectorstore, retriever, llm, template


# Initialize RAG
try:
    vectorstore, retriever, llm, template = load_rag()
    rag_available = True
except Exception as e:
    rag_available = False
    st.error(f"Failed to load RAG system: {e}")


# Initialize Chat Messages
if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding:10px;">
            <div style="font-size:45px;">🧠</div>
            <h2 style="margin-bottom:0;">RAG Assistant</h2>
            <p style="color:#94a3b8; font-size:13px;">Your intelligent document assistant</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### ⚙️ System Status")
    if rag_available:
        st.success("🟢 RAG System Online")
    else:
        st.error("🔴 RAG System Offline")

    st.divider()

    # Upload Documents
    st.markdown("### 📤 Upload PDF / Book")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="pdf_uploader",
        help="Upload books, notes, or PDFs to parse and index into ChromaDB"
    )

    docs_dir = "document loaders"

    if uploaded_files:
        if st.button("⚡ Process & Index Documents", use_container_width=True, type="primary"):
            if not rag_available:
                st.error("RAG system offline. Please verify vector database and API keys.")
            else:
                os.makedirs(docs_dir, exist_ok=True)
                total_chunks = 0
                processed_files = []
                with st.spinner("Parsing & embedding uploaded file(s)..."):
                    for u_file in uploaded_files:
                        save_path = os.path.join(docs_dir, u_file.name)
                        with open(save_path, "wb") as f:
                            f.write(u_file.getbuffer())

                        docs = []
                        if u_file.name.endswith(".pdf"):
                            try:
                                loader = PyPDFLoader(save_path)
                                docs = loader.load()
                            except Exception as err:
                                st.error(f"Error loading PDF '{u_file.name}': {err}")
                        elif u_file.name.endswith(".txt"):
                            try:
                                loader = TextLoader(save_path, encoding="utf-8")
                                docs = loader.load()
                            except Exception as err:
                                st.error(f"Error loading TXT '{u_file.name}': {err}")

                        if docs:
                            splitter = RecursiveCharacterTextSplitter(
                                chunk_size=1000,
                                chunk_overlap=200
                            )
                            chunks = splitter.split_documents(docs)
                            if chunks:
                                vectorstore.add_documents(chunks)
                                total_chunks += len(chunks)
                                processed_files.append(u_file.name)

                if total_chunks > 0:
                    st.cache_resource.clear()
                    st.success(f"Successfully processed {len(processed_files)} file(s) into {total_chunks} vector chunks!")
                    st.rerun()

    st.divider()

    # Document List
    st.markdown("### 📚 Loaded Documents")
    if os.path.exists(docs_dir):
        doc_files = [f for f in os.listdir(docs_dir) if f.endswith((".pdf", ".txt"))]
        if doc_files:
            for f in doc_files:
                col_name, col_del = st.columns([0.82, 0.18])
                with col_name:
                    icon = "📄" if f.endswith(".pdf") else "📝"
                    st.markdown(f"{icon} `{f}`")
                with col_del:
                    if st.button("🗑️", key=f"del_{f}", help=f"Delete {f} from DB and disk"):
                        # 1. Purge from ChromaDB
                        if rag_available:
                            try:
                                data = vectorstore._collection.get()
                                ids_to_del = []
                                for idx, meta in enumerate(data.get("metadatas", [])):
                                    src = meta.get("source", "")
                                    if f.lower() in src.lower() or os.path.basename(src).lower() == f.lower():
                                        ids_to_del.append(data["ids"][idx])
                                if ids_to_del:
                                    vectorstore._collection.delete(ids=ids_to_del)
                            except Exception as err:
                                st.error(f"Error purging ChromaDB: {err}")

                        # 2. Remove file from disk
                        file_path = os.path.join(docs_dir, f)
                        if os.path.exists(file_path):
                            try:
                                os.remove(file_path)
                            except Exception as err:
                                st.error(f"Error deleting file: {err}")

                        # 3. Clear cache and refresh
                        st.cache_resource.clear()
                        st.success(f"Deleted '{f}' and purged vectors!")
                        st.rerun()
        else:
            st.info("No documents found in `document loaders/`.")
    else:
        st.warning("Folder `document loaders/` missing.")

    st.divider()

    st.markdown("### 💬 Controls")
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col_btn2:
        if st.button("🔄 Reload DB", use_container_width=True):
            st.cache_resource.clear()
            st.success("Vector DB Reloaded!")
            st.rerun()


# ============================================================
# HERO HEADER & STATUS DASHBOARD
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🧠 RAG AI Assistant</div>
        <div class="hero-subtitle">
            Ask questions about your uploaded documents and get grounded answers powered by ChromaDB & Mistral AI.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="status-card"><div class="status-title">STATUS</div><div class="status-value online">● Online</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="status-card"><div class="status-title">MODEL</div><div class="status-value">Mistral Small</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="status-card"><div class="status-title">VECTOR DB</div><div class="status-value">ChromaDB</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="status-card"><div class="status-title">RETRIEVAL</div><div class="status-value">Hybrid Semantic</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# WELCOME SCREEN
# ============================================================

if len(st.session_state.messages) == 0:
    st.markdown(
        """
        <div style="text-align:center; padding:15px 0;">
            <h2>How can I help you today?</h2>
            <p style="color:#94a3b8;">Ask questions grounded in your document collection.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="welcome-card"><div class="welcome-icon">📚</div><div class="welcome-title">Learn from PDFs & Notes</div><div class="welcome-text">Ask anything from your uploaded PDF guides and text notes.</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="welcome-card"><div class="welcome-icon">🔎</div><div class="welcome-title">Semantic Retrieval</div><div class="welcome-text">Similarity search fetches the most relevant text passages.</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="welcome-card"><div class="welcome-icon">🤖</div><div class="welcome-title">Grounded AI Answers</div><div class="welcome-text">Mistral AI generates answers strictly from source text.</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🧠"):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander(f"📚 View Sources ({len(message['sources'])})"):
                for i, source in enumerate(message["sources"], start=1):
                    source_name = source.get("source", "Document")
                    page = source.get("page", None)
                    page_text = f" · Page {page + 1}" if page is not None else ""
                    
                    html_card = f'<div class="source-card"><div class="source-title">📄 Source {i}</div><div class="source-text">{source_name}{page_text}</div></div>'
                    st.markdown(html_card, unsafe_allow_html=True)


# ============================================================
# PROCESS QUERY
# ============================================================

query = st.chat_input("Ask something about your documents...")

if query:
    if not rag_available:
        st.error("RAG system offline. Please check your database connection.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user", avatar="👤"):
        st.markdown(query)

    with st.chat_message("assistant", avatar="🧠"):
        with st.spinner("🔎 Searching your documents..."):
            try:
                # Smart Retrieval: Match document filename if user mentions it or key terms
                clean_q = query.lower()
                matched_docs = []

                if os.path.exists(docs_dir):
                    for f in os.listdir(docs_dir):
                        if f.endswith((".pdf", ".txt")):
                            fname_lower = f.lower()
                            fname_no_ext = os.path.splitext(f)[0].lower()
                            if (fname_no_ext in clean_q or fname_lower in clean_q or
                                ("resume" in clean_q and "resume" in fname_lower) or
                                ("vanshika" in clean_q and "vanshika" in fname_lower) or
                                ("dsa" in clean_q and "dsa" in fname_lower) or
                                ("notes" in clean_q and "notes" in fname_lower)):
                                try:
                                    all_data = vectorstore._collection.get()
                                    for idx, meta in enumerate(all_data.get("metadatas", [])):
                                        src = meta.get("source", "")
                                        if f.lower() in src.lower():
                                            doc_obj = CustomDoc(
                                                page_content=all_data["documents"][idx],
                                                metadata=meta
                                            )
                                            matched_docs.append(doc_obj)
                                except Exception:
                                    pass

                if matched_docs:
                    docs = matched_docs[:4]
                else:
                    docs = retriever.invoke(query)

                context = "\n\n".join([doc.page_content for doc in docs])
                final_prompt = template.invoke({"context": context, "question": query})

                response = llm.invoke(final_prompt)
                answer = response.content

                st.markdown(answer)

                sources = []
                for doc in docs:
                    metadata = doc.metadata or {}
                    sources.append({
                        "source": metadata.get("source", "Unknown document"),
                        "page": metadata.get("page", None)
                    })

                if sources:
                    with st.expander(f"📚 View Sources ({len(sources)})"):
                        for i, source in enumerate(sources, start=1):
                            source_name = source["source"]
                            page = source["page"]
                            page_text = f" · Page {page + 1}" if page is not None else ""
                            
                            html_card = f'<div class="source-card"><div class="source-title">📄 Source {i}</div><div class="source-text">{source_name}{page_text}</div></div>'
                            st.markdown(html_card, unsafe_allow_html=True)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as e:
                error_msg = f"An error occurred: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg, "sources": []})