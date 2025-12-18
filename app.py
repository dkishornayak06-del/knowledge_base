__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import streamlit as st
import os
import tempfile
from pypdf import PdfReader
from fastembed import TextEmbedding
import chromadb
from chromadb.config import Settings
from groq import Groq

st.set_page_config(page_title="RAG Chatbot", layout="wide")

api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("GROQ_API_KEY not configured")
    st.stop()

client = Groq(api_key=api_key)

@st.cache_resource
def load_resources():
    embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    db_dir = os.path.join(tempfile.gettempdir(), "chroma_rag_db")
    chroma_client = chromadb.PersistentClient(
        path=db_dir,
        settings=Settings(anonymized_telemetry=False)
    )
    return embedder, chroma_client

embedder, chroma_client = load_resources()

def get_collection():
    return chroma_client.get_or_create_collection(
        name="rag_collection",
        metadata={"hnsw:space": "cosine"}
    )

with st.sidebar:
    st.subheader("Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )
    process_btn = st.button("Process & Train")

if process_btn and uploaded_files:
    status = st.empty()
    status.info("Processing documents...")

    try:
        chroma_client.delete_collection("rag_collection")
    except:
        pass

    collection = get_collection()
    chunks = []

    for file in uploaded_files:
        text = ""
        try:
            if file.name.endswith(".pdf"):
                reader = PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() or ""
            else:
                text = file.read().decode("utf-8")

            chunk_size = 800
            overlap = 100
            for i in range(0, len(text), chunk_size - overlap):
                chunk = text[i:i + chunk_size]
                if len(chunk) > 50:
                    chunks.append(chunk)
        except:
            pass

    if chunks:
        for i in range(0, len(chunks), 100):
            batch = chunks[i:i + 100]
            embeddings = [e.tolist() for e in embedder.embed(batch)]
            ids = [f"doc_{i+j}" for j in range(len(batch))]
            collection.add(documents=batch, embeddings=embeddings, ids=ids)

        status.success(f"Indexed {len(chunks)} chunks")
    else:
        status.error("No readable text found")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

prompt = st.chat_input("Ask a question")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    try:
        collection = get_collection()
        q_embed = embedder.embed([prompt])[0].tolist()
        results = collection.query(query_embeddings=[q_embed], n_results=5)

        if results["documents"] and results["documents"][0]:
            context = "\n".join(results["documents"][0])[:8000]
            final_prompt = (
                "Answer using only the context below.\n\n"
                f"Context:\n{context}\n\nQuestion:\n{prompt}"
            )

            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": final_prompt}],
                max_tokens=500
            )
            answer = response.choices[0].message.content
        else:
            answer = "No relevant information found."

    except Exception:
        answer = "Rate limit reached. Please try again later."

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)
