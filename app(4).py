import os
import hashlib
from typing import List, Dict, Tuple

import fitz  # PyMuPDF
import faiss
import numpy as np
import streamlit as st
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# App configuration
# ============================================================

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide",
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-20b"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 5


# ============================================================
# Helpers
# ============================================================

def get_groq_api_key() -> str:
    """Read the API key from Streamlit secrets or environment variables."""
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        key = ""

    if not key:
        key = os.environ.get("GROQ_API_KEY", "")

    return key


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model() -> SentenceTransformer:
    """
    Load an open-source embedding model.
    SentenceTransformer performs tokenization internally before
    generating embeddings.
    """
    return SentenceTransformer(EMBEDDING_MODEL)


def extract_pdf_text(pdf_bytes: bytes) -> List[Dict]:
    """Extract text from every PDF page and keep page numbers."""
    pages = []

    document = fitz.open(stream=pdf_bytes, filetype="pdf")

    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text").strip()

        if text:
            pages.append(
                {
                    "page": page_number,
                    "text": text,
                }
            )

    document.close()
    return pages


def create_chunks(pages: List[Dict]) -> List[Dict]:
    """
    Split page text into overlapping chunks while preserving
    the original PDF page number.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []

    for page_data in pages:
        page_chunks = splitter.split_text(page_data["text"])

        for chunk_index, chunk in enumerate(page_chunks):
            clean_chunk = chunk.strip()

            if clean_chunk:
                chunks.append(
                    {
                        "text": clean_chunk,
                        "page": page_data["page"],
                        "chunk_id": chunk_index,
                    }
                )

    return chunks


def build_faiss_index(
    chunks: List[Dict],
    embedding_model: SentenceTransformer,
) -> Tuple[faiss.Index, np.ndarray]:
    """
    Generate normalized embeddings and store them in a FAISS
    inner-product index. With normalized vectors, inner product
    behaves like cosine similarity.
    """
    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index, embeddings


def retrieve_chunks(
    question: str,
    index: faiss.Index,
    chunks: List[Dict],
    embedding_model: SentenceTransformer,
    top_k: int = TOP_K,
) -> List[Dict]:
    """Retrieve the most relevant chunks for a question."""
    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    number_to_retrieve = min(top_k, len(chunks))

    scores, indices = index.search(
        question_embedding,
        number_to_retrieve,
    )

    retrieved = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        item = dict(chunks[idx])
        item["score"] = float(score)
        retrieved.append(item)

    return retrieved


def create_context(retrieved_chunks: List[Dict]) -> str:
    """Create a source-labelled context block for the LLM."""
    context_parts = []

    for number, chunk in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"[Source {number} | PDF page {chunk['page']}]\n"
            f"{chunk['text']}"
        )

    return "\n\n---\n\n".join(context_parts)


def generate_answer(
    question: str,
    retrieved_chunks: List[Dict],
    api_key: str,
) -> str:
    """Ask the Groq-hosted open-weight model to answer from retrieved context."""
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    context = create_context(retrieved_chunks)

    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the supplied PDF context.

Rules:
1. Do not invent information that is not supported by the context.
2. If the answer is not present in the context, clearly say:
   "I couldn't find that information in the uploaded PDF."
3. Give a concise but useful explanation.
4. When making a factual claim from the context, cite the PDF page
   using the format [Page X].
5. Do not cite sources that were not provided in the context.

UPLOADED PDF CONTEXT:
{context}

USER QUESTION:
{question}
"""

    response = client.responses.create(
        model=GROQ_MODEL,
        input=prompt,
    )

    return response.output_text


def file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


# ============================================================
# Session state
# ============================================================

if "index" not in st.session_state:
    st.session_state.index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "document_name" not in st.session_state:
    st.session_state.document_name = ""

if "document_hash" not in st.session_state:
    st.session_state.document_hash = ""

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# UI
# ============================================================

st.title("PDF RAG Assistant")
st.write(
    "Upload a PDF, build a FAISS vector index, and ask questions "
    "using an open-weight model served through the Groq API."
)

with st.sidebar:
    st.header("1. Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF document",
        type=["pdf"],
        help="The PDF is processed in memory for the current app session.",
    )

    st.divider()

    st.header("RAG Configuration")
    st.write(f"**Embedding model:** `{EMBEDDING_MODEL}`")
    st.write(f"**LLM:** `{GROQ_MODEL}`")
    st.write(f"**Chunk size:** {CHUNK_SIZE} characters")
    st.write(f"**Chunk overlap:** {CHUNK_OVERLAP} characters")
    st.write(f"**Top-k retrieval:** {TOP_K}")

    st.divider()

    if st.session_state.index is not None:
        st.success(
            f"Indexed {len(st.session_state.chunks)} chunks "
            f"from `{st.session_state.document_name}`."
        )


if uploaded_file is not None:
    pdf_bytes = uploaded_file.getvalue()
    current_hash = file_hash(pdf_bytes)

    # Build the index only when a new PDF is uploaded.
    if current_hash != st.session_state.document_hash:
        with st.status("Processing PDF...", expanded=True) as status:
            try:
                st.write("Extracting PDF text...")
                pages = extract_pdf_text(pdf_bytes)

                if not pages:
                    st.error(
                        "No selectable text was found in this PDF. "
                        "Scanned/image-only PDFs require OCR."
                    )
                    st.stop()

                st.write(f"Extracted text from {len(pages)} page(s).")

                st.write("Creating overlapping chunks...")
                chunks = create_chunks(pages)

                if not chunks:
                    st.error("No text chunks could be created.")
                    st.stop()

                st.write(f"Created {len(chunks)} chunks.")

                st.write("Generating embeddings...")
                embedding_model = load_embedding_model()

                st.write("Building FAISS vector index...")
                index, _ = build_faiss_index(
                    chunks,
                    embedding_model,
                )

                st.session_state.index = index
                st.session_state.chunks = chunks
                st.session_state.document_name = uploaded_file.name
                st.session_state.document_hash = current_hash
                st.session_state.messages = []

                status.update(
                    label="PDF indexed successfully.",
                    state="complete",
                    expanded=False,
                )

            except Exception as exc:
                status.update(
                    label="PDF processing failed.",
                    state="error",
                    expanded=True,
                )
                st.exception(exc)

if st.session_state.index is None:
    st.info(
        "Upload a text-based PDF from the sidebar. "
        "The app will extract, chunk, embed, and index it automatically."
    )
    st.stop()


# ============================================================
# Chat history
# ============================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


question = st.chat_input(
    "Ask a question about the uploaded PDF..."
)

if question:
    api_key = get_groq_api_key()

    if not api_key:
        st.error(
            "GROQ_API_KEY is not configured. Add it to Streamlit Secrets "
            "before asking questions."
        )
        st.stop()

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the PDF and generating an answer..."):
            try:
                embedding_model = load_embedding_model()

                retrieved = retrieve_chunks(
                    question=question,
                    index=st.session_state.index,
                    chunks=st.session_state.chunks,
                    embedding_model=embedding_model,
                    top_k=TOP_K,
                )

                answer = generate_answer(
                    question=question,
                    retrieved_chunks=retrieved,
                    api_key=api_key,
                )

                st.markdown(answer)

                with st.expander("Retrieved sources"):
                    for source_number, item in enumerate(retrieved, start=1):
                        st.markdown(
                            f"**Source {source_number} — PDF page "
                            f"{item['page']} — similarity "
                            f"{item['score']:.3f}**"
                        )
                        st.write(item["text"])

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

            except Exception as exc:
                error_message = (
                    "The answer could not be generated. "
                    "Please check your Groq API key and try again."
                )
                st.error(error_message)
                st.exception(exc)
