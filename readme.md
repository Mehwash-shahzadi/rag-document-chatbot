# 🤖 Customer Support Knowledge Base Assistant

A powerful, private chatbot that lets you upload your own documents (PDF/TXT) and ask questions using state-of-the-art language models. Built for customer support teams, this assistant leverages retrieval-augmented generation (RAG) to provide accurate, context-aware answers from your knowledge base.

---

## 🚀 Features

- **Document Ingestion:** Upload and process PDF or TXT files to build your searchable knowledge base.
- **Semantic Search:** Uses advanced embeddings and FAISS vector store for fast, relevant document retrieval.
- **Conversational Chatbot:** Natural, context-aware Q&A powered by HuggingFace LLMs (Mistral-7B-Instruct).
- **Interactive Web UI:** Streamlit-based chat interface with chat history, confidence badges, and feedback.
- **Feedback & Improvement:** Collects user feedback for continuous improvement and tracks helpfulness.

---

## 🛠️ Tech Stack

- [LangChain](https://github.com/langchain-ai/langchain) (document loading, splitting, RAG)
- [HuggingFace Transformers](https://huggingface.co/) (embeddings & LLMs)
- [FAISS](https://github.com/facebookresearch/faiss) (vector database)
- [Streamlit](https://streamlit.io/) (web interface)
- Python 3.9+

---

## ⚡ Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/customer-support-chatbot.git
   cd customer-support-chatbot
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv .venv
   # On Windows:
   .\.venv\Scripts\activate
   # On Mac/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirement.txt
   ```

4. **Set up your `.env` file**
   - Copy `.env.example` to `.env` (or create `.env`)
   - Add your HuggingFace API token (get it from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens))
   ```env
   HUGGINGFACE_API_TOKEN=your_huggingface_token_here
   ```

---

## 📝 Usage

1. **Start the Streamlit app**

   ```bash
   streamlit run app.py
   ```

2. **Upload your documents**

   - Use the sidebar to upload PDF or TXT files.
   - Click "Process Documents" to add them to your knowledge base.

3. **Chat with your assistant**
   - Ask questions in the chat box.
   - Get answers with confidence scores and source citations.
   - Provide feedback with thumbs up/down.

---

## 📁 Project Structure

```
customer-support-chatbot/
│
├── app.py                  # Streamlit web app
├── .env                    # Environment variables (API keys, config)
├── requirement.txt         # Python dependencies
├── config/
│   └── settings.py         # Configuration loader
├── src/
│   ├── document_processor.py  # Document loading & splitting
│   ├── embeddings.py          # Embedding generation
│   ├── vector_store.py        # FAISS vector store management
│   ├── retriever.py           # Semantic search retriever
│   ├── chatbot.py             # Main chatbot logic
│   └── utils.py               # Utility functions
└── ...
```

---

## ⚙️ Configuration Options

Edit your `.env` file to customize:

- `HUGGINGFACE_API_TOKEN`: Your HuggingFace API key (**required**)
- `EMBEDDING_MODEL`: Embedding model (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `LLM_MODEL`: LLM for answering (default: `mistralai/Mistral-7B-Instruct-v0.2`)
- `CHUNK_SIZE`: Text chunk size for splitting (default: `500`)
- `CHUNK_OVERLAP`: Overlap between chunks (default: `50`)
- `TOP_K`: Number of top results to retrieve (default: `3`)
- `VECTOR_DB_PATH`: Path to vector database (default: `./vector_db`)
- `RAW_DOCS_PATH`: Path to raw documents (default: `./docs`)

---

## 🐞 Troubleshooting

- **Missing HuggingFace Token:**  
  Make sure `HUGGINGFACE_API_TOKEN` is set in your `.env` file.

- **Virtual Environment Issues:**  
  Activate your virtual environment before running commands.

- **Vector Store Not Initialized:**  
  Upload and process at least one document before chatting.

- **API Rate Limits:**  
  Free HuggingFace accounts have rate limits. Upgrade or retry later if you see errors.

- **File Upload Errors:**  
  Only PDF and TXT files are supported. Check file size and format.

---

## 🌱 Future Enhancements

- 🔒 User authentication and multi-user support
- 🗂️ Support for more file types (Word, HTML, etc.)
- 📊 Admin dashboard for analytics and feedback review
- 🧠 Fine-tuning and custom LLM support
- ☁️ Cloud deployment templates (Docker, Azure, AWS)

---

## 📸 Screenshots

> ![Chat UI Screenshot](screenshots/chat-ui.png) > ![Sidebar Screenshot](screenshots/sidebar.png)

---
