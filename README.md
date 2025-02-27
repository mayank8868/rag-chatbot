📄 Chat with your PDF (Llama 2 RAG)

This is a Retrieval-Augmented Generation (RAG) chatbot built with Llama 2, allowing users to upload PDFs and ask questions based on the document content. It extracts text, converts it into vector embeddings using FAISS, and retrieves relevant sections for generating AI-powered responses.

🚀 Features
✅ Upload PDFs and extract text automatically
✅ Retrieve relevant document chunks using FAISS
✅ AI-powered answers using Llama 2
✅ Streamlit UI for easy interaction
✅ GPU acceleration for faster responses (CUDA/PyTorch)

📌 Installation
1️⃣ Clone the repository

git clone https://github.com/mayank8868/rag-chatbot
cd chat-with-pdf-llama2

2️⃣ Set up a virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate  # On Windows

3️⃣ Install dependencies

pip install -r requirements.txt

📥 Download Llama 2 Model
This project requires Llama 2 (7B Chat) GGUF model. Download it from:
🔗 Meta AI Official Website

After downloading, place the model file in the models/ directory:

models/llama-2-7b-chat.Q4_K_M.gguf
🏃‍♂️ Running the Application

1️⃣ Process PDFs (Create Vector Store)
Run the PDF processing script to generate embeddings:
python pdf_processing.py

2️⃣ Start the Streamlit Web App

streamlit run scripts/app.py
Access the chatbot at: http://localhost:8501

📜 File Structure

📂 chat-with-pdf-llama2
│-- 📂 data
│   │-- 📂 saved_pdfs          # Uploaded PDFs
│   │-- 📂 vector_store        # FAISS vector database
│-- 📂 models                  # Llama 2 model files
│-- app.py                     # Streamlit UI for chatbot
│-- chatbot.py                 # Chatbot logic with Llama 2
│-- pdf_processing.py           # Extract text & create FAISS embeddings
│-- requirements.txt            # Dependencies list
│-- README.md                   # Project documentation
⚙️ Configuration
Modify the configuration inside app.py and chatbot.py as needed.

Parameter	Description
MODEL_PATH	Path to the Llama 2 model file
Vector Store Directory	data/vector_store/ for FAISS embeddings
GPU Acceleration	Uses CUDA if available (n_gpu_layers=-1 in Llama config)
🛠️ Troubleshooting
❌ Model file not found error
Ensure the Llama 2 model file is inside the models/ directory.

🛠️ CUDA device not found
Check if PyTorch with CUDA is installed:


python -c "import torch; print(torch.cuda.is_available())"
If False, install the correct version of PyTorch:


pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
📌 Future Enhancements
✅ Multi-PDF support
✅ Chat history persistence
✅ Optimize response time

🤝 Contributing
Feel free to open an issue or submit a pull request! 🚀

📜 License
This project is licensed under the MIT License.
