import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import tempfile

# PAGE CONFIG
st.set_page_config(page_title="AI PDF QA System", layout="wide")

# 🌈 SUNSET THEME CSS
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #2c1a4d, #6a3093, #ff416c, #ff9a44);
    color: white;
}
h1 {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    background: linear-gradient(90deg, #ff9a44, #ff416c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

h3 {
    text-align: center;
    color: #f1c4e0;
}

.stButton > button {
    background: linear-gradient(90deg, #ff416c, #ff9a44);
    color: white;
    font-size: 18px;
    padding: 12px 25px;
    border-radius: 12px;
    border: none;
    transition: 0.3s;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #ff4b2b, #ff6a00);
    transform: scale(1.05);
}

/* Force input text color ALWAYS visible */
input[type="text"] {
    color: black !important;
    background-color: rgba(255,255,255,0.2) !important;
}

/* When typing */
input[type="text"]:focus {
    color: black !important;
    background-color: rgba(255,255,255,0.25) !important;
}

/* Cursor color */
input[type="text"] {
    caret-color: black !important;
}

/* Placeholder */
input[type="text"]::placeholder {
    color: #eeeeee !important;
}

/* Labels */
label {
    color: black !important;
}

.answer-box {
    background: rgba(255,255,255,0.15);
    padding: 20px;
    border-radius: 15px;
    font-size: 18px;
    backdrop-filter: blur(10px);
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>

/* File uploader text */
.stFileUploader label {
    color: white !important;
    font-size: 18px;
}

/* Text input label */
.stTextInput label {
    color: white !important;
    font-size: 18px;
}

</style>
""", unsafe_allow_html=True)


# SESSION STATE
if "page" not in st.session_state:
    st.session_state.page = "welcome"

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "qa_pipeline" not in st.session_state:
    st.session_state.qa_pipeline = None


# LOAD MODEL 

@st.cache_resource
def load_model():
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    return pipe

if st.session_state.qa_pipeline is None:
    st.session_state.qa_pipeline = load_model()

qa_pipeline = st.session_state.qa_pipeline

if st.session_state.page == "welcome":

    st.markdown("<h1>🤖 AI PDF QA System</h1>", unsafe_allow_html=True)
    st.markdown("<h3>Ask questions from your PDF easily</h3>", unsafe_allow_html=True)

    # Create 3 columns to center button
    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        if st.button("🚀 Start", key="start_btn", use_container_width=True):
            st.session_state.page = "upload"
            st.rerun()


# PAGE 2: UPLOAD PDF

elif st.session_state.page == "upload":

    st.title("📂 Upload Your PDF")

    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    if uploaded_file is not None and st.session_state.vectorstore is None:

        with st.spinner("📖 Processing PDF..."):

            # Save temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                pdf_path = tmp.name

            loader = PyPDFLoader(pdf_path)
            pages = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=150,
                chunk_overlap=20
            )

            texts = splitter.split_documents(pages)
            texts = texts[:50]  # limit for speed

            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )

            vectorstore = FAISS.from_documents(texts, embeddings)

            st.session_state.vectorstore = vectorstore

        st.success("✅ PDF processed!")

    if st.session_state.vectorstore is not None:
        if st.button("➡️ Go to Question Page", key="next_btn"):
            st.session_state.page = "qa"
            st.rerun()


# PAGE 3: QA

elif st.session_state.page == "qa":

    st.title("💬 Ask Questions from PDF")

    query = st.text_input("Enter your question:")

    def get_answer(query):
        docs = st.session_state.vectorstore.similarity_search(query, k=2)
        context = " ".join([doc.page_content for doc in docs])[:800]

        if len(context.strip()) < 50:
            return "Not found in document", context

        prompt = f"""
You are a helpful assistant.

Read the context carefully and answer the question properly in 1-2 complete sentences.

Do NOT give single word answers.
Do NOT repeat the question.
If answer is not found, say "Not found in document".

Context:
{context}

Question: {query}

Answer:
"""

        response = qa_pipeline(
        prompt,
        max_new_tokens=100,
        min_new_tokens=30
        )
        answer = response[0]['generated_text'].replace(prompt, "").strip()
        if len(answer.split()) < 3:
            return context[:200], context

        if answer == "":
            answer = "Not found in document"

        return answer, context

    if st.button("🔍 Get Answer", key="answer_btn"):

        if query.strip() == "":
            st.warning("⚠️ Please enter a question")

        else:
            with st.spinner("🤔 Thinking..."):
                answer, context = get_answer(query)

            st.markdown(f"""
            <div class="answer-box">
                ✅ {answer}
            </div>
            """, unsafe_allow_html=True)

            with st.expander("📚 View Context"):
                st.write(context)

    if st.button("⬅️ Upload New PDF", key="back_btn"):
        st.session_state.page = "upload"
        st.session_state.vectorstore = None
        st.rerun()