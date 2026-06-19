# PDF Analyzer

A RAG (Retrieval-Augmented Generation) app that lets you chat with any PDF using Gemini and FAISS.

---

## Stack

- Gemini 2.5 Flash — LLM
- Gemini Embedding 001 — embeddings
- FAISS — vector store
- LangChain — RAG pipeline
- Streamlit — UI

---

## Setup

1. Clone the repo and install dependencies:

```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the root folder:

```
GOOGLE_API_KEY=your_key_here
```

3. Run the app:

```bash
streamlit run app.py
```

---

## Usage

- Upload any PDF using the file uploader
- Ask questions about its content in the chat
- The app only answers from the document, not outside knowledge
- The FAISS index is saved locally so re-uploading the same PDF skips re-embedding

---

## Project Structure

```
.
├── app.py               # Streamlit app
├── PDF_analyzer.ipynb           # Original notebook (experimentation)
├── requirements.txt
├── .env                 # Not committed
└── README.md
```

---

## Notes

- Free tier API keys are limited to 100 embedding requests per minute. Large PDFs are embedded in batches with a delay to avoid hitting this limit.
- The FAISS index is saved as a folder named `<pdf_name>_faiss_index` in the project root.
