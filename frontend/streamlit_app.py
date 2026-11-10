
import sys
from pathlib import Path

import streamlit as st

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.pipeline import ResearchPipeline


st.set_page_config(
    page_title="Evidence-Based AI Research Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("Evidence-Based AI Research Assistant")
st.caption(
    "Upload research papers, ask questions, "
    "and get AI-generated answers backed by evidence."
)

# Initialize pipeline once per Streamlit session
if "pipeline" not in st.session_state:
    with st.spinner("Initializing AI models..."):
        st.session_state.pipeline = ResearchPipeline()

pipeline = st.session_state.pipeline


# --------------------------------------------------
# 1. PDF Upload and Ingestion
# --------------------------------------------------

st.subheader("1. Upload Research Paper")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
)

if uploaded_file is not None:

    if st.button("Ingest PDF", type="primary"):

        temp_dir = PROJECT_ROOT / "data" / "uploads"
        temp_dir.mkdir(parents=True, exist_ok=True)

        pdf_path = temp_dir / uploaded_file.name
        pdf_path.write_bytes(uploaded_file.getvalue())

        try:
            with st.spinner("Processing and indexing research paper..."):
                chunk_count = pipeline.ingest_pdf(pdf_path)

            st.success(
                f"Successfully ingested {uploaded_file.name}. "
                f"Created {chunk_count} chunks."
            )

        except Exception as exc:
            st.error(f"PDF ingestion failed: {exc}")


# --------------------------------------------------
# 2. Research Question
# --------------------------------------------------

st.divider()

st.subheader("2. Ask a Research Question")

question = st.text_input(
    "Enter your question",
    placeholder="What does the research paper say about machine learning?",
)

top_k = st.slider(
    "Number of evidence chunks",
    min_value=1,
    max_value=10,
    value=3,
)

if st.button("Generate Answer", type="primary"):

    if not question.strip():
        st.warning("Please enter a question.")

    else:
        try:
            with st.spinner(
                "Retrieving evidence and generating your answer..."
            ):
                result = pipeline.ask(
                    question=question,
                    top_k=top_k,
                )

            # --------------------------------------
            # 3. Generated Answer
            # --------------------------------------

            st.divider()
            st.subheader("3. Generated Answer")

            st.markdown(result["answer"])

            # --------------------------------------
            # 4. Supporting Evidence
            # --------------------------------------

            evidence = result["evidence"]

            if evidence:
                st.divider()
                st.subheader("4. Supporting Evidence")

                for i, (chunk, score) in enumerate(
                    evidence,
                    start=1,
                ):
                    source = chunk.get(
                        "source",
                        chunk.get("filename", "Unknown"),
                    )

                    page = chunk.get(
                        "page_number",
                        chunk.get("page", "Unknown"),
                    )

                    with st.expander(
                        f"Evidence {i} | Similarity: {score:.3f}",
                        expanded=(i == 1),
                    ):
                        st.write(chunk["text"])

                        st.caption(
                            f"Source: {source} | Page: {page}"
                        )

            else:
                st.info("No supporting evidence was retrieved.")

        except Exception as exc:
            st.error(f"Answer generation failed: {exc}")


# --------------------------------------------------
# 5. Sidebar
# --------------------------------------------------

with st.sidebar:
    st.title("About NoteWise")

    st.write(
        """
        An evidence-based AI research assistant
        powered by:

        - PDF text extraction
        - Sentence Transformers
        - FAISS vector search
        - Groq LLM
        - Retrieval-Augmented Generation (RAG)
        """
    )

    st.divider()

    st.caption("Built with Python and Streamlit.")
