"""
Week 2 · Day 4 — Multi-PDF RAG App with Sources

Run with:
    export OPENAI_API_KEY="your-key"
    streamlit run rag_app.py

Upload several PDFs at once -> they are chunked, embedded and stored in ONE
FAISS vector store (kept in st.session_state). Every answer lists its sources
(file name + page number) taken from the chunk metadata.
"""

import os
import subprocess
import sys
import tempfile

import streamlit as st
from streamlit import runtime

# If this file was started with plain `python rag_app.py` (e.g. VS Code's
# Run button), relaunch it properly with `streamlit run` so the browser UI
# and st.session_state work.
if __name__ == "__main__" and not runtime.exists():
    sys.exit(subprocess.call(
        [sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)]
        + sys.argv[1:]
    ))

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4

st.set_page_config(page_title="Multi-PDF RAG", page_icon="📚")
st.title("📚 Multi-PDF RAG with Sources")

# API key comes from the environment only — never hard-coded.
if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is not set. Run `export OPENAI_API_KEY=...` "
             "in your terminal, then restart Streamlit.")
    st.stop()


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------
def build_store(uploaded_files):
    """Load every PDF, split into chunks, embed all into one FAISS store."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    all_chunks = []
    per_file = {}

    for f in uploaded_files:
        # PyPDFLoader needs a path, so write the upload to a temp file.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.getvalue())
            tmp_path = tmp.name

        try:
            pages = PyPDFLoader(tmp_path).load()  # one Document per page
        finally:
            os.remove(tmp_path)

        # The loader sets 'source' to the temp path — replace it with the
        # real file name so citations are readable. 'page' is kept as-is
        # and survives the splitter into each chunk's metadata.
        for p in pages:
            p.metadata["source"] = f.name

        chunks = splitter.split_documents(pages)
        per_file[f.name] = len(chunks)
        all_chunks.extend(chunks)

    if not all_chunks:
        return None, 0, per_file

    store = FAISS.from_documents(all_chunks, OpenAIEmbeddings())
    return store, len(all_chunks), per_file


files = st.file_uploader(
    "Upload PDFs", type="pdf", accept_multiple_files=True
)

# Re-index only when the set of uploaded files changes.
file_key = tuple(sorted((f.name, f.size) for f in files)) if files else None

if files and st.session_state.get("file_key") != file_key:
    with st.spinner(f"Indexing {len(files)} PDF(s)..."):
        store, n_chunks, per_file = build_store(files)
    st.session_state.store = store
    st.session_state.n_chunks = n_chunks
    st.session_state.per_file = per_file
    st.session_state.file_key = file_key
    st.session_state.history = []

if not files:
    st.info("Upload one or more PDFs to get started.")
    st.stop()

if st.session_state.get("store") is None:
    st.warning("No text could be extracted from these PDFs "
               "(scanned images need OCR first).")
    st.stop()

st.write("**Uploaded:** " + ", ".join(f.name for f in files))
st.success(f"Indexed {st.session_state.get('n_chunks', 0)} chunks.")
with st.expander("Chunks per file"):
    for name, n in st.session_state.get("per_file", {}).items():
        st.write(f"- {name}: {n} chunks")


# ---------------------------------------------------------------------------
# Question answering
# ---------------------------------------------------------------------------
def page_label(doc):
    # PyPDFLoader pages are 0-indexed; show humans 1-indexed page numbers.
    return int(doc.metadata.get("page", 0)) + 1


def format_context(docs):
    blocks = []
    for d in docs:
        tag = f"[{d.metadata['source']} - page {page_label(d)}]"
        blocks.append(f"{tag}\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)


PROMPT = """You answer questions using ONLY the context below, which comes
from several different PDF files. Each excerpt starts with a tag like
[file.pdf - page N].

Rules:
- Cite the tag of every excerpt you rely on, inline, e.g. (handbook.pdf - page 7).
- If different documents give different or conflicting answers, do NOT pick
  one silently. Say clearly that the sources disagree and state what each
  one says, with its citation.
- If the context does not contain the answer, say "I couldn't find that in
  the uploaded documents." Do not use outside knowledge.

Context:
{context}

Question: {question}

Answer:"""

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
retriever = st.session_state.store.as_retriever(search_kwargs={"k": TOP_K})

if "history" not in st.session_state:
    st.session_state.history = []


def show_sources(docs):
    # De-duplicate (file, page) pairs while keeping retrieval order.
    seen, labels = set(), []
    for d in docs:
        key = (d.metadata["source"], page_label(d))
        if key not in seen:
            seen.add(key)
            labels.append(f"{key[0]} - page {key[1]}")
    st.caption("**Sources:** " + " | ".join(labels))

    with st.expander("Show retrieved chunks"):
        for d in docs:
            st.markdown(f"**{d.metadata['source']} - page {page_label(d)}**")
            st.text(d.page_content[:800])


# Replay earlier Q&A so the conversation stays on screen.
for item in st.session_state.history:
    with st.chat_message("user"):
        st.write(item["question"])
    with st.chat_message("assistant"):
        st.write(item["answer"])
        show_sources(item["docs"])

question = st.chat_input("Ask a question across all your PDFs")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            docs = retriever.invoke(question)
            answer = llm.invoke(
                PROMPT.format(context=format_context(docs), question=question)
            ).content
        st.write(answer)
        show_sources(docs)

    st.session_state.history.append(
        {"question": question, "answer": answer, "docs": docs}
    )