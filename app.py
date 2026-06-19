

# --------------------------------------------------------------------------------------------------------------------------
# Libraries

import streamlit as st
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
 
import os
import re
import tempfile
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# --------------------------------------------------------------------------------------------------------------------------
# Page config
st.set_page_config(
    page_title="PDF Analyzer",
    page_icon="📄",
    layout="centered"
)

st.title("PDF Analyzer")
st.caption("Upload a PDF and talk")

# --------------------------------------------------------------------------------------------------------------------------
# Preprocessing
def preprocess_document(text) -> str:
    text = re.sub(r'Page \d+\s*(?:of|\/)\s*\d+', '', text)
    text = re.sub(r'—{2,}|_{3,}', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_query(query) -> str:
    return " ".join(query.split()).strip()

# --------------------------------------------------------------------------------------------------------------------------
# Prompt
TEMPLATE = """
You are a helpful assistant that answers questions strictly based on the provided document context.

Rules:
- Answer from the context below. Do NOT use outside knowledge.
- If the answer is not within context, clearly say so.
- Be concise but complete. Cite the relevant part when useful.

Context:
{context}

Question: {question}

Answer:
"""

prompt = PromptTemplate(
    template=TEMPLATE,
    input_variables=["context", "question"]
)

# --------------------------------------------------------------------------------------------------------------------------
# Cached loaders 
@st.cache_resource
def load_model():
    return GoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

@st.cache_resource
def load_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
@st.cache_resource(show_spinner="Processing PDF...")
def build_qa_chain(pdf_bytes: bytes, pdf_name: str):
    # Write uploaded bytes to a temp file so PyPDFLoader can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    for doc in documents:
        doc.page_content = preprocess_document(doc.page_content)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)

    embeddings = load_embeddings()

    index_path = Path(pdf_name).stem + "_faiss_index"
    if Path(index_path).exists():
        vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(index_path)

    qa_chain = RetrievalQA.from_chain_type(
        llm=load_model(),
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt}
    )

    os.unlink(tmp_path)
    return qa_chain, len(documents)

# --------------------------------------------------------------------------------------------------------------------------
# UI 
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    pdf_bytes = uploaded_file.read()
    pdf_name  = uploaded_file.name

    try:
        qa_chain, num_pages = build_qa_chain(pdf_bytes, pdf_name)
        st.success(f"Ready — {num_pages} pages loaded from **{pdf_name}**")
    except Exception as e:
        st.error(f"Failed to process PDF: {e}")
        st.stop()

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    if user_input := st.chat_input("Ask something about the PDF..."):
        query = preprocess_query(user_input)

        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = qa_chain.invoke({"query": query})
                    answer = result["result"]
                except Exception as e:
                    answer = f"Error: {e}"

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("Upload a PDF above to get started.")
