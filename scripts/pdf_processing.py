import os
from pathlib import Path
import fitz  # PyMuPDF
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Define directories
TMP_DIR = Path("data/tmp")
VECTOR_STORE_DIR = Path("data/vector_store")
TMP_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# Define vector store file path
FAISS_INDEX_PATH = VECTOR_STORE_DIR / "faiss_index.pkl"

def load_new_documents():
    """Loads new PDFs and returns their text content."""
    pdf_files = list(TMP_DIR.glob("*.pdf"))
    new_texts = {}

    for file in pdf_files:
        faiss_file = VECTOR_STORE_DIR / f"{file.stem}.faiss"
        if faiss_file.exists():
            print(f"✅ Skipping already processed: {file.name}")
            continue  # Skip already processed PDFs

        with fitz.open(file) as doc:
            text = "\n".join([page.get_text("text") for page in doc])
            if text.strip():
                new_texts[file.stem] = text

    return new_texts

def update_vector_store(new_texts):
    """Updates FAISS vector store with new document embeddings."""
    if not new_texts:
        print("🚀 No new PDFs to process.")
        return

    # Load existing FAISS index if it exists
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    if FAISS_INDEX_PATH.exists():
        with open(FAISS_INDEX_PATH, "rb") as f:
            vector_store = pickle.load(f)
        print("🔄 Loaded existing FAISS index.")
    else:
        vector_store = FAISS(embeddings)
        print("🆕 Created new FAISS index.")

    # Process new texts
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    for doc_name, text in new_texts.items():
        split_texts = text_splitter.split_text(text)
        doc_vectors = FAISS.from_texts(split_texts, embeddings)
        
        # Add new vectors to existing FAISS index
        vector_store.merge_from(doc_vectors)
        
        # Mark this PDF as processed
        with open(VECTOR_STORE_DIR / f"{doc_name}.faiss", "wb") as f:
            pickle.dump(doc_vectors, f)

    # Save updated FAISS index
    with open(FAISS_INDEX_PATH, "wb") as f:
        pickle.dump(vector_store, f)
    print("✅ FAISS index updated successfully.")

# Run processing
if __name__ == "__main__":
    new_texts = load_new_documents()
    update_vector_store(new_texts)
