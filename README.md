# 🤖 AI PDF Question Answering System

An intelligent **RAG (Retrieval-Augmented Generation)** application that allows users to upload PDF documents and ask questions in natural language. The system retrieves the most relevant content using semantic search and generates context-aware answers using transformer-based NLP models.

---

# 🚀 Features

✅ Upload any text-based PDF
✅ Ask questions directly from document content
✅ Semantic search using embeddings
✅ Context-aware answer generation
✅ Interactive and modern Streamlit UI
✅ Fast document retrieval with FAISS
✅ Real-world NLP + LLM implementation

---

# 🧠 Technologies Used

* Python
* Streamlit
* LangChain
* FAISS Vector Database
* Hugging Face Transformers
* FLAN-T5 Model
* Sentence Embeddings

---

# ⚙️ System Workflow

```text
PDF Upload
   ↓
Text Extraction
   ↓
Text Chunking
   ↓
Embeddings Generation
   ↓
FAISS Vector Storage
   ↓
User Query
   ↓
Semantic Retrieval
   ↓
LLM Answer Generation
```

---

# 📌 How It Works

1. User uploads a PDF document.
2. The system extracts and cleans text.
3. Text is divided into smaller chunks.
4. Embeddings are generated for semantic understanding.
5. FAISS stores embeddings for fast similarity search.
6. User asks a question.
7. Relevant chunks are retrieved.
8. FLAN-T5 generates a context-based answer.

---

# 🖥️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/ai-pdf-qa.git
cd ai-pdf-qa
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
streamlit run ui.py
```

---


# 🎯 Project Highlights

* Demonstrates practical implementation of RAG architecture
* Uses semantic similarity instead of keyword search
* Combines retrieval + generation for improved accuracy
* Beginner-friendly UI with advanced NLP backend

---

# ⚠️ Limitations

* Works best with text-based PDFs
* Small language model may generate short answers
* Accuracy depends on retrieved context quality

---

# 🔮 Future Improvements

* Multi-PDF support
* OCR for scanned PDFs
* Chat history
* Better LLM integration
* Cloud deployment

---

# 👩‍💻 Author

**Mehak Garg**

AI/ML | NLP | Generative AI Enthusiast

---

# ⭐ If you like this project

Give this repository a star and connect for more AI & NLP projects.
