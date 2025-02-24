import streamlit as st  # ✅ Must be first Streamlit command
import os
import pickle
import pdfplumber
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import LlamaCpp

# ✅ Set Streamlit Page Configuration
st.set_page_config(page_title="📄 Chat with your PDF (Llama 2 RAG)", layout="wide")
# Add Title
st.title("📄 Chat with your PDF (Llama 2 RAG)")
# Custom CSS for Styling
st.markdown("""
    <style>
        .stTextInput>div>div>input { font-size: 18px; }
        .stButton>button { border-radius: 8px; font-size: 18px; }
        .stMarkdown { font-size: 16px; }
        .chat-container { border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: #f9f9f9; }
        .user-message { background-color: #A1E3F9; color: black; padding: 10px; border-radius: 10px; margin-bottom: 10px; }
        .ai-message { background-color: #71BBB2; color: white; padding: 10px; border-radius: 10px; margin-bottom: 10px; }
        [data-testid="stSidebar"] { background-color: rgb(255, 246, 188); }
        [data-testid="stAppViewContainer"] { background-color: rgb(176, 221, 254); }
    </style>
""", unsafe_allow_html=True)

# ✅ Initialize session state
if "qa_chain" not in st.session_state:
    st.session_state["qa_chain"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# 🔹 Sidebar - Model Configuration
st.sidebar.header("⚙️ Model Configuration")
MODEL_PATH = "models/llama-2-7b-chat.Q4_K_M.gguf"

if not os.path.exists(MODEL_PATH):
    st.sidebar.error(f"❌ Model file not found: {MODEL_PATH}")
    st.stop()
else:
    st.sidebar.success("✅ Llama Model Loaded!")

# 🔹 Sidebar - Upload PDF
st.sidebar.header("📂 Upload Your PDF")
SAVED_PDF_DIR = "data/saved_pdfs"
os.makedirs(SAVED_PDF_DIR, exist_ok=True)

uploaded_file = st.sidebar.file_uploader("Drag and drop a PDF file", type=["pdf"])

# 🔹 Sidebar - Select existing PDFs
saved_pdfs = ["None"] + [f for f in os.listdir(SAVED_PDF_DIR) if f.endswith(".pdf")]
selected_pdf = st.sidebar.selectbox("📂 Select an existing PDF", saved_pdfs)

file_path = None
if uploaded_file:
    file_path = os.path.join(SAVED_PDF_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success(f"📂 File {uploaded_file.name} uploaded successfully!")

elif selected_pdf and selected_pdf != "None":
    file_path = os.path.join(SAVED_PDF_DIR, selected_pdf)

# 🔹 Process PDF Button
if file_path and st.sidebar.button("🚀 Process PDF"):
    faiss_path = file_path + ".faiss"

    with st.spinner("🔍 Processing the document..."):
        if os.path.exists(faiss_path):
            with open(faiss_path, "rb") as f:
                vector_store = pickle.load(f)
        else:
            with pdfplumber.open(file_path) as pdf:
                text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

            if not text.strip():
                st.sidebar.error("❌ No extractable text found in the PDF.")
                st.stop()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
            split_docs = text_splitter.split_text(text)

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vector_store = FAISS.from_texts(split_docs, embeddings)

            with open(faiss_path, "wb") as f:
                pickle.dump(vector_store, f)

        retriever = vector_store.as_retriever(search_kwargs={"k": 5, "search_type": "mmr"})

        llm = LlamaCpp(
            model_path=MODEL_PATH, 
            n_gpu_layers=-1 if torch.cuda.is_available() else 0,  # Use GPU if available
            n_ctx=4096,
            f16_kv=True,
            streaming=True,
            verbose=False,
            temperature=1.0
        )

        qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

    st.success("✅ Document processed successfully!")
    st.session_state["qa_chain"] = qa_chain
    st.session_state["chat_history"] = []  # ✅ Reset chat history on new document processing

# 🔹 Chat Interface
query = st.text_input("Type your question here...", key="user_query")

if query and st.session_state["qa_chain"]:
    with st.spinner("🤖 Thinking..."):
        response = st.session_state["qa_chain"].invoke({"query": query})  # ✅ Fixed incorrect `.run()` call

    if response and "result" in response:
        st.session_state["chat_history"].append({"query": query, "response": response["result"]})
    else:
        st.warning("🤖 AI could not generate a response. Try another question.")

# ✅ Safely Iterate Over Chat History
if "chat_history" in st.session_state and isinstance(st.session_state["chat_history"], list):
    for chat in st.session_state["chat_history"]:
        st.markdown(f'**You:** {chat["query"]}')
        st.markdown(f'**AI:** {chat["response"]}')
