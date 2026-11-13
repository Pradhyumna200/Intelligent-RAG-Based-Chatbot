import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings

# Project paths
BASE_DIR = Path(__file__).resolve().parent
VECTOR_DB_PATH = BASE_DIR / "vector_db"

# Load Google Gemini API key
load_dotenv(BASE_DIR / ".env")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

st.set_page_config(
    page_title="Customer Support RAG Chatbot",
    page_icon="💬",
    layout="centered"
)

st.title("💬 Customer Support Chatbot")
st.write("Ask a question about customer support.")

if not GOOGLE_API_KEY:
    st.error("Google API key not found. Please check your .env file.")
    st.stop()

@st.cache_resource
def load_rag_pipeline():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma(
        collection_name="customer_support",
        persist_directory=str(VECTOR_DB_PATH),
        embedding_function=embeddings
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    llm = ChatGoogleGenerativeAI(
       model="gemini-3.6-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2
    )

    prompt = ChatPromptTemplate.from_template("""
    You are a helpful customer support assistant.
    Answer using only the context provided.
    If the context does not contain enough information,
    say that you do not have enough information.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": lambda inputs: format_docs(
                retriever.invoke(inputs["question"])
            ),
            "question": lambda inputs: inputs["question"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

try:
    rag_chain = load_rag_pipeline()
except Exception as error:
    st.error(f"Could not initialize the chatbot: {error}")
    st.stop()

question = st.text_input(
    "Enter your question:",
    placeholder="How can I contact customer support?"
)

if st.button("Get Answer"):
    if question.strip():
        with st.spinner("Finding relevant information..."):
            try:
                answer = rag_chain.invoke({"question": question})
                st.subheader("Chatbot Answer")
                st.write(answer)
            except Exception as error:
                st.error(f"Unable to generate an answer: {error}")
    else:
        st.warning("Please enter a question.")