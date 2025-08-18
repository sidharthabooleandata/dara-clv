
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import re
import requests
import time
from typing import List, Dict

# ---------------- CONFIG ----------------
TXT_PATH = r"unstructured_finance_interactions 1.txt"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"   # try "paraphrase-MiniLM-L3-v2" if still slow
TOP_K_DEFAULT = 8
CHUNK_SIZE = 5000
CHUNK_OVERLAP = 120
MAX_CONTEXT_CHARS = 2800


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

OPENROUTER_MODEL = "mistralai/mistral-7b-instruct:free"

UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I
)

# ---------------- Helpers ----------------
def read_text(path=TXT_PATH) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def normalize_newlines(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\r", "\n")

def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[Dict]:
    text = normalize_newlines(text).strip()
    if not text:
        return []
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    chunks = []
    id_counter = 0
    for b in blocks:
        if len(b) <= chunk_size:
            chunks.append({"id": id_counter, "text": b})
            id_counter += 1
            continue
        start = 0
        while start < len(b):
            end = min(start + chunk_size, len(b))
            chunk = b[start:end]
            chunks.append({"id": id_counter, "text": chunk})
            id_counter += 1
            if end == len(b):
                break
            start = max(0, end - overlap)
    return chunks

@st.cache_resource(show_spinner=False)
def load_embedding_model(name=EMBED_MODEL_NAME):
    return SentenceTransformer(name)

@st.cache_resource(show_spinner=False)
def get_index_and_model(raw_text):
    """Build FAISS index + embeddings once and cache it."""
    chunks = chunk_text(raw_text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    embed_model = load_embedding_model()
    texts = [c["text"] for c in chunks]
    embs = embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs_norm = embs / norms
    d = embs_norm.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embs_norm)
    return chunks, embed_model, index, embs_norm

def retrieve_by_embedding(query: str, _embed_model, index, chunks: List[Dict], embs_norm: np.ndarray, top_k=5) -> List[Dict]:
    q_emb = _embed_model.encode([query], convert_to_numpy=True)
    q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)
    scores, ids = index.search(q_emb, top_k)
    ids = ids[0].tolist()
    scores = scores[0].tolist()
    results = []
    for idx, sc in zip(ids, scores):
        if idx < 0 or idx >= len(chunks):
            continue
        results.append({"chunk_id": chunks[idx]["id"], "score": float(sc), "text": chunks[idx]["text"]})
    return results

def generate_answer(question: str, retrieved_chunks: List[Dict]):
    # ✅ truncate context so it doesn’t overload OpenRouter
    context_text = ""
    total_len = 0
    for c in retrieved_chunks:
        if total_len + len(c["text"]) > MAX_CONTEXT_CHARS:
            break
        context_text += "\n\n" + c["text"]
        total_len += len(c["text"])

    prompt = f"""You are a helpful assistant. Use the retrieved context to answer the question.
If the answer is not fully in the context, you may also use your general knowledge.

Context:
{context_text}

Question: {question}
Answer:
"""
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": OPENROUTER_MODEL,
            "messages": [{"role": "user", "content": prompt}],
        }
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", json=data, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error generating answer: {e}"

# ---------------- UI ----------------
st.set_page_config(layout="wide", page_title="💬 RAG Chat — OpenRouter Free Model")

# Load file
raw_text = read_text(TXT_PATH)
if not raw_text:
    st.error(f"File not found: {TXT_PATH}")
    st.stop()

# Build index once (cached)
start = time.time()
chunks, embed_model, index, embs_norm = get_index_and_model(raw_text)
st.sidebar.write(f"✅ Index loaded in {time.time()-start:.2f}s")

# Sidebar
with st.sidebar:
    st.image(
        "https://booleandata.com/wp-content/uploads/2022/09/Boolean-logo_Boolean-logo-USA-1.png",
        use_container_width=True
    )

# Conversation area
st.subheader("Customer Lifetime Value Prediction")
if "history" not in st.session_state:
    st.session_state.history = []
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

for msg in st.session_state.history:
    if msg["role"] == "user":
        st.markdown(f"**🧑 You:** {msg['text']}")
    else:
        st.markdown(f"**🤖 Boolean Assistant:** {msg['text']}")

# Fixed bottom input
st.markdown("""
    <style>
    [data-testid="stChatInput"] {
        position: fixed;
        bottom: 0;
        left: 250px; /* start after sidebar width */
        right: 0;
        background: white;
        padding: 8px 10px;
        border-top: 1px solid #ddd;
        z-index: 999;
    }
    [data-testid="stChatInput"] textarea {
        padding-right: 36px !important;
    }
    .send-icon {
        position: absolute;
        right: 20px;
        bottom: 20px;
        width: 20px;
        height: 20px;
    }
    </style>
""", unsafe_allow_html=True)

user_query = st.chat_input("Type your question...")

st.markdown(
    '<img class="send-icon" src="https://cdn-icons-png.flaticon.com/512/724/724715.png">',
    unsafe_allow_html=True
)

# Handle input
if user_query:
    st.session_state.history.append({"role": "user", "text": user_query})
    retrieved = retrieve_by_embedding(user_query, embed_model, index, chunks, embs_norm, top_k=TOP_K_DEFAULT)
    st.session_state.last_retrieved = retrieved
    answer = generate_answer(user_query, retrieved)
    st.session_state.history.append({"role": "assistant", "text": answer})
    st.rerun()

