
import sys
from pathlib import Path

import streamlit as st

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.retriever import Retriever


st.set_page_config(
    page_title="Evidence-Based AI Research Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("Evidence-Based AI Research Assistant")
st.caption("Upload research papers, ask questions, and retrieve supporting evidence.")

# Initialize Retriever once per Streamlit session
if "retriever" not in st.session_state:
    with st.spinner("Initializing AI models..."):
        st.session_state.retriever = Retriever()

retriever = st.session_state.retriever

# PDF upload
st.subheader("1. Upload Research Paper")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
)

if uploaded_file is not None:
    if st.button("Ingest PDF"):
        temp_dir = PROJECT_ROOT / "data" / "uploads"
        temp_dir.mkdir(parents=True, exist_ok=True)

        pdf_path = temp_dir / uploaded_file.name
        pdf_path.write_bytes(uploaded_file.getvalue())

        try:
            with st.spinner("Processing research paper..."):
                chunk_count = retriever.ingest_pdf(pdf_path)

            st.success(
                f"Successfully ingested {uploaded_file.name}. "
                f"Created {chunk_count} chunks."
            )

        except Exception as exc:
            st.error(f"PDF ingestion failed: {exc}")

# Question answering through evidence retrieval
st.divider()
st.subheader("2. Ask a Research Question")

question = st.text_input(
    "Enter your question",
    placeholder="What does this research paper say about machine learning?",
)

if st.button("Find Evidence"):
    if not question.strip():
        st.warning("Please enter a question.")

    else:
        try:
            with st.spinner("Searching relevant evidence..."):
                results = retriever.retrieve(question, top_k=3)

            if not results:
                st.info("No relevant evidence found.")

            else:
                st.subheader("Retrieved Evidence")

                for i, (chunk, score) in enumerate(results, start=1):
                    with st.expander(
                        f"Evidence {i} | Similarity: {score:.3f}",
                        expanded=True,
                    ):
                        st.write(chunk["text"])
                        st.caption(
                            f"Source: {chunk['source']} | "
                            f"Page: {chunk['page_number']}"
                        )

        except Exception as exc:
            st.error(f"Retrieval failed: {exc}")
