import os
import tempfile
import shutil
from typing import List, TypedDict

import streamlit as st

# --- OpenAI via Cornell gateway (picked up from env) ---
#   OPENAI_API_KEY, OPENAI_BASE_URL must be set
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Loaders / docs / splitters
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector store (handle both import paths)
try:
    from langchain_chroma import Chroma
except Exception:  # fallback for some environments
    from langchain_community.vectorstores import Chroma

# Prompt + parsing
from langchain_core.prompts import PromptTemplate

# LangGraph
from langgraph.graph import START, StateGraph


# ---------------- UI ----------------
st.set_page_config(page_title="RAG Q&A", page_icon="📄", layout="wide")
st.title("📄 RAG Q&A (txt + pdf)")

with st.sidebar:
    uploads = st.file_uploader("Upload files", type=["txt", "md", "pdf"], accept_multiple_files=True)
    chunk_size = st.slider("Chunk size", 200, 2000, 800, step=50)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, 120, step=10)
    top_k = st.slider("Top-K chunks", 1, 10, 4)
    rebuild = st.button("Rebuild index")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Upload documents and ask a question."}]
if "retriever_ready" not in st.session_state:
    st.session_state.retriever_ready = False
if "persist_dir" not in st.session_state:
    st.session_state.persist_dir = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None


# ------------- Helpers -------------
def _reset_store():
    if st.session_state.persist_dir and os.path.exists(st.session_state.persist_dir):
        shutil.rmtree(st.session_state.persist_dir, ignore_errors=True)
    st.session_state.persist_dir = None
    st.session_state.vectorstore = None
    st.session_state.retriever_ready = False


def _load_docs(files) -> List[Document]:
    if not files:
        return []
    tmp = tempfile.mkdtemp()
    docs: List[Document] = []

    for f in files or []:
        name = f.name
        lower = name.lower()
        if lower.endswith((".txt", ".md")):
            # .getvalue() avoids EOF issues on rebuilds
            text = f.getvalue().decode("utf-8", errors="ignore")
            if text.strip():
                docs.append(Document(page_content=text, metadata={"source": name}))
        elif lower.endswith(".pdf"):
            path = os.path.join(tmp, name)
            with open(path, "wb") as out:
                out.write(f.getvalue())

            # First try PyPDF; if it yields no text, try PyMuPDF
            pages = []
            try:
                pages = PyPDFLoader(path).load()
            except Exception:
                pages = []

            if not any((p.page_content or "").strip() for p in pages):
                try:
                    from langchain_community.document_loaders import PyMuPDFLoader
                    pages = PyMuPDFLoader(path).load()
                except Exception:
                    pages = []

            for d in pages:
                if (d.page_content or "").strip():
                    d.metadata.update({"source": name})
                    docs.append(d)

    return docs

def _files_signature(files):
    return tuple((f.name, getattr(f, "size", None), f.type) for f in (files or []))

if "files_sig" not in st.session_state:
    st.session_state.files_sig = ()


def _chunk(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


def _ensure_index():
    # Rebuild when user clicks button
    if rebuild:
        _reset_store()

    # Rebuild automatically when the uploaded files change
    current_sig = _files_signature(uploads)
    if st.session_state.files_sig != current_sig:
        _reset_store()
        st.session_state.files_sig = current_sig

    # If already built, nothing to do
    if st.session_state.retriever_ready:
        return

    # Build from current uploads
    docs = _load_docs(uploads)
    # Drop empty docs defensively
    docs = [d for d in docs if (d.page_content or "").strip()]
    if not docs:
        return

    chunks = _chunk(docs)
    st.session_state.persist_dir = tempfile.mkdtemp()

    # Embeddings: pass env explicitly to avoid “connection error”
    embeddings = OpenAIEmbeddings(
        model="openai.text-embedding-3-large",
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.ai.it.cornell.edu"),
    )

    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=st.session_state.persist_dir,
        collection_name="rag_docs",
    )
    st.session_state.vectorstore = vs
    st.session_state.retriever_ready = True



def _format_docs(docs: List[Document]) -> str:
    out = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", None)
        tag = f"{src}{(' p.' + str(page)) if page is not None else ''}"
        snippet = d.page_content.replace("\n", " ")
        out.append(f"[{tag}] {snippet}")
    return "\n\n".join(out[:8])


# ------------- LangGraph RAG chain -------------
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


def retrieve(state: State) -> dict:
    docs = st.session_state.vectorstore.similarity_search(state["question"], k=top_k)
    return {"context": docs}


template = (
    "You are an assistant for question-answering. Use ONLY the context.\n"
    "If the answer is not in the context, say you don't know.\n\n"
    "Question: {question}\n\nContext:\n{context}\n\nAnswer:"
)
prompt = PromptTemplate.from_template(template)
llm = ChatOpenAI(model="openai.gpt-4o", temperature=0)

def generate(state: State) -> dict:
    ctx = _format_docs(state["context"])
    msg = prompt.invoke({"question": state["question"], "context": ctx})
    resp = llm.invoke(msg)
    return {"answer": resp.content}


graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()


# ------------- Chat UI -------------
for m in st.session_state.messages[-20:]:
    st.chat_message(m["role"]).write(m["content"])

question = st.chat_input("Ask a question about your documents", disabled=uploads is None or len(uploads) == 0)

if question:
    _ensure_index()
    if not st.session_state.retriever_ready:
        st.warning("Upload at least one .txt or .pdf to build the index.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        st.chat_message("user").write(question)

        with st.chat_message("assistant"):
            result = graph.invoke({"question": question})
            answer = result["answer"]
            st.write(answer)

            # Sources
            st.markdown("**Top sources:**")
            for i, d in enumerate(result["context"]):
                src = d.metadata.get("source", "unknown")
                page = d.metadata.get("page", None)
                st.caption(f"{i+1}. {src}{('  p.' + str(page)) if page is not None else ''}")

        st.session_state.messages.append({"role": "assistant", "content": answer})

with st.sidebar:
    if st.session_state.retriever_ready:
        # quick peek at what made it in
        try:
            # crude count by source
            from collections import Counter
            # pull a small sample to avoid heavy calls
            sample = st.session_state.vectorstore._collection.get(include=["metadatas"], limit=200)
            sources = [m.get("source", "unknown") for m in (sample.get("metadatas") or [])]
            counts = Counter(sources)
            st.write("**Indexed chunks by source:**")
            for src, c in counts.most_common():
                st.caption(f"{src}: {c}")
        except Exception:
            pass
