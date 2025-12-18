# app.py
# Fully working Streamlit app: PDF/TXT upload + summarization + Q&A using Groq

import streamlit as st
import os
import time
from groq import Groq
from PyPDF2 import PdfReader

# -------------------- CONFIG --------------------
st.set_page_config(page_title="Groq PDF Assistant", layout="wide")

# Load API key
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY not configured")
    st.stop()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama3-8b-8192"

# -------------------- HELPERS --------------------

def read_file(file):
    if file.type == "application/pdf":
        reader = PdfReader(file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        return file.read().decode("utf-8")


def chunk_text(text, chunk_size=1500, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def groq_call(messages, max_tokens=300):
    for attempt in range(3):
        try:
            return client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=max_tokens
            ).choices[0].message.content
        except Exception as e:
            if "rate limit" in str(e).lower():
                time.sleep(5)
            else:
                raise e

# -------------------- SESSION STATE --------------------
if "docs" not in st.session_state:
    st.session_state.docs = []
if "summary" not in st.session_state:
    st.session_state.summary = ""

# -------------------- UI --------------------
st.sidebar.title("📚 Knowledge Base")
files = st.sidebar.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

process = st.sidebar.button("Process & Train")

st.title("Groq PDF Assistant")

# -------------------- PROCESS FILES --------------------
if process and files:
    st.session_state.docs.clear()
    with st.spinner("Processing documents..."):
        for file in files:
            text = read_file(file)
            chunks = chunk_text(text)
            st.session_state.docs.extend(chunks)
    st.success(f"Processed {len(st.session_state.docs)} chunks")

# -------------------- SUMMARIZE --------------------
if st.button("Summarise") and st.session_state.docs:
    with st.spinner("Generating summary..."):
        joined = "\n".join(st.session_state.docs[:5])  # limit chunks
        st.session_state.summary = groq_call([
            {"role": "system", "content": "Summarize the following document."},
            {"role": "user", "content": joined}
        ])

if st.session_state.summary:
    st.subheader("📄 Summary")
    st.write(st.session_state.summary)

# -------------------- Q&A --------------------
st.subheader("❓ Ask a Question")
question = st.text_input("Ask a question")

if st.button("Ask") and question and st.session_state.docs:
    with st.spinner("Thinking..."):
        context = "\n".join(st.session_state.docs[:4])
        answer = groq_call([
            {"role": "system", "content": "Answer using the context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ])
    st.markdown("### ✅ Answer")
    st.write(answer)

# -------------------- FOOTER --------------------
st.caption("Powered by Groq • Streamlit")
