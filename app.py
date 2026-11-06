
import os

import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import build_vector_store , ask_question

load_dotenv()

st.set_page_config(
    page_title="Chat with Your PDFs",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Chat with Your PDFs")
st.caption("Ask questions and get answers grounded in your documents.")

if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is missing. Add it to your .env file.")
    st.stop()

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Your documents")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button(
        "Process PDFs",
        type="primary",
        disabled=not uploaded_files,
    ):
        try:
            with st.spinner(
                "Extracting text, creating embeddings, and indexing..."
            ):
                vectorstore, page_count, chunk_count = (
                    build_vector_store(uploaded_files)
                )

            st.session_state.vectorstore = vectorstore
            st.session_state.messages = []

            st.success("PDFs processed successfully!")
            st.write(f"Pages with text: {page_count}")
            st.write(f"Text chunks indexed: {chunk_count}")

        except Exception as error:
            st.error(f"Could not process PDFs: {error}")

    if st.button("Clear chat"):
        st.session_state.messages = []

    if st.button("Remove indexed documents"):
        st.session_state.vectorstore = None
        st.session_state.messages = []
        st.rerun()

if st.session_state.vectorstore is None:
    st.info("Upload one or more PDFs in the sidebar and click Process PDFs.")
    st.stop()

# Display the existing conversation.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("View retrieved sources"):
                for source in message["sources"]:
                    st.markdown(
                        f"**{source['source']} — Page {source['page']}**"
                    )
                    st.caption(source["content"][:1500])

# Accept a new question.
question = st.chat_input("Ask a question about your PDFs...")

if question:
    previous_history = st.session_state.messages.copy()

    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching your PDFs and generating an answer..."):
                answer, sources = ask_question(
                    st.session_state.vectorstore,
                    question,
                    previous_history,
                )

            st.markdown(answer)

            if sources:
                with st.expander("View retrieved sources"):
                    for source in sources:
                        st.markdown(
                            f"**{source['source']} — Page {source['page']}**"
                        )
                        st.caption(source["content"][:1500])

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })

        except Exception as error:
            st.error(f"Could not generate an answer: {error}")
            st.session_state.messages.pop()
