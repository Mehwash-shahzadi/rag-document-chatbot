# 🤖 Customer Support Knowledge Base Assistant

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License](https://img.shields.io/badge/License-MIT-green)

A **private, intelligent chatbot** that empowers customer support teams to query their own documents securely.  
Built with **retrieval-augmented generation (RAG)**, it provides **accurate, context-aware responses** from your internal knowledge base — no external data leakage.

---

## About

This assistant is designed for teams and businesses who want to leverage their own documentation for instant, reliable answers.  
**All data stays local**—your files are never sent to third-party servers.  
_The bot responds after a short delay (about 2 seconds) to simulate natural, human-like conversation._

---

## 🚀 Features

- 📄 **Document Ingestion:** Upload and process PDF or TXT files to create your searchable knowledge base.
- 🔍 **Semantic Search:** Retrieve precise answers using FAISS and transformer-based embeddings.
- 💬 **Conversational Chatbot:** Natural, context-aware responses powered by HuggingFace LLMs (Mistral-7B-Instruct).
- 🖥️ **Interactive Web UI:** Streamlit interface with history, confidence badges, and feedback tools.
- 📈 **Continuous Improvement:** Collect feedback for retraining or knowledge optimization.

---

## 🛠️ Tech Stack

- [LangChain](https://github.com/langchain-ai/langchain) – Document processing & RAG pipeline
- [HuggingFace Transformers](https://huggingface.co/) – Embeddings & LLMs
- [FAISS](https://github.com/facebookresearch/faiss) – Vector similarity search
- [Streamlit](https://streamlit.io/) – Web UI
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
   # Windows
   .\.venv\Scripts\activate
   # Mac/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your `.env` file**

   ```bash
   cp .env.example .env
   ```

   **Example:**

   ```env
   HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
   EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
   LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.2
   ```

---

## 🧠 Usage

1. **Run the Streamlit app**

   ```bash
   streamlit run app.py
   ```

2. **Upload your documents**

   - Use the sidebar to upload `.pdf` or `.txt` files.
   - Click **“Process Documents”** to add them to your knowledge base.

3. **Chat naturally**

   - Ask questions and get context-based answers.
   - View **confidence scores** and **source citations**.
   - Give **feedback** (👍 / 👎) to improve responses.
   - _Note: The bot responds after a short delay (about 2 seconds) to simulate a natural conversation._

---

## 📁 Project Structure

```
customer-support-chatbot/
│
├── app.py                     # Streamlit web app
├── .env                       # Environment variables
├── requirements.txt           # Python dependencies
├── config/
│   └── settings.py            # Configuration loader
├── src/
│   ├── document_processor.py  # Document ingestion & text splitting
│   ├── embeddings.py          # Embedding generation
│   ├── vector_store.py        # FAISS vector database management
│   ├── retriever.py           # Document retrieval logic
│   ├── chatbot.py             # LLM interaction and response generation
│   └── utils.py               # Helper functions
└── ...
```

---

## ⚙️ Configuration

Customize your `.env` file:

| Variable                   | Description               | Default                                  |
| -------------------------- | ------------------------- | ---------------------------------------- |
| `HUGGINGFACEHUB_API_TOKEN` | HuggingFace API key       | —                                        |
| `EMBEDDING_MODEL`          | Model for embeddings      | `sentence-transformers/all-MiniLM-L6-v2` |
| `LLM_MODEL`                | Model for chat responses  | `mistralai/Mistral-7B-Instruct-v0.2`     |
| `CHUNK_SIZE`               | Text chunk size           | `500`                                    |
| `CHUNK_OVERLAP`            | Overlap between chunks    | `50`                                     |
| `TOP_K`                    | Top results for retrieval | `3`                                      |
| `VECTOR_DB_PATH`           | Vector store directory    | `./vector_db`                            |
| `RAW_DOCS_PATH`            | Raw document storage      | `./docs`                                 |

---

## 🐞 Troubleshooting

| Issue                  | Solution                                        |
| ---------------------- | ----------------------------------------------- |
| **Missing Token**      | Set `HUGGINGFACEHUB_API_TOKEN` in your `.env`   |
| **Virtual Env Issues** | Ensure `.venv` is activated                     |
| **Vector DB Empty**    | Upload and process at least one document        |
| **Rate Limits**        | HuggingFace free tiers have limits; retry later |
| **File Upload Errors** | Only PDF/TXT files are supported                |

---

## 🌱 Future Enhancements

- 🔒 User authentication & multi-user sessions
- 🗂️ Additional file formats (DOCX, HTML, Markdown)
- 📊 Admin analytics dashboard
- 🧠 Fine-tuning & custom model integration
- ☁️ Cloud deployment templates (Docker, AWS, Azure)

---

## 📸 Screenshots

| Chat Interface                      | Sidebar                             |
| ----------------------------------- | ----------------------------------- |
| ![Chat UI](screenshots/chat-ui.png) | ![Sidebar](screenshots/sidebar.png) |

---

## 🤝 Contributing

Contributions are welcome!  
Please open an issue or submit a pull request. See the `CONTRIBUTING.md` (coming soon) for guidelines.

---

**Made with ❤️ using LangChain & Streamlit**
