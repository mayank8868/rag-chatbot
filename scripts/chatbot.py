import torch
import streamlit as st
from llama_cpp import Llama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os

# ✅ Set device to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ✅ Load Llama 2 with GPU Acceleration
MODEL_PATH = "models/llama-2-7b-chat.Q4_K_M.gguf"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found: {MODEL_PATH}")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,  # Increase context length
    n_gpu_layers=-1,  # Use GPU for all layers
    verbose=False
    
)

# ✅ Load FAISS vector store with embeddings
VECTOR_STORE_DIR = "data/vector_store"

if not os.path.exists(VECTOR_STORE_DIR):
    raise FileNotFoundError(f"❌ Vector store not found at {VECTOR_STORE_DIR}. Run pdf_processing.py first.")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.load_local(VECTOR_STORE_DIR, embeddings, allow_dangerous_deserialization=True)
retriever = vector_store.as_retriever()

# ✅ Memory for conversation history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ✅ Chatbot UI
def chatbot_ui():
    st.title("🤖 Llama 2 RAG Chatbot (GPU Accelerated)")

    # Display previous messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message("user").write(message["user"])
        st.chat_message("assistant").write(message["assistant"])

    # User input
    user_input = st.chat_input("Ask a question...")
    if user_input:
        st.chat_message("user").write(user_input)

        # Retrieve documents
        relevant_docs = retriever.invoke(user_input)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # ✅ Create Llama 2 prompt
        prompt = f"Context:\n{context}\n\nUser: {user_input}\nLlama:"
        response = llm(prompt)["choices"][0]["text"]

        # ✅ Display response
        st.chat_message("assistant").write(response)

        # ✅ Store chat history
        st.session_state.messages.append({"user": user_input, "assistant": response})

# Run the chatbot UI
if __name__ == "__main__":
    chatbot_ui()
