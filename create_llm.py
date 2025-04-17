# create_vectorstore.py

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# ✅ Step 1: Load PDFs from folder
DATA_PATH = "data/"

def load_pdf_files(data_path):
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()

documents = load_pdf_files(DATA_PATH)

# ✅ Step 2: Split text into chunks
def create_chunks(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

text_chunks = create_chunks(documents)
print(f"✅ Total Chunks Created: {len(text_chunks)}")

# ✅ Step 3: Generate embeddings
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embedding_model = get_embedding_model()

# ✅ Step 4: Store embeddings in FAISS
DB_FAISS_PATH = "vectorstore/db_faiss"
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)
print(f"✅ Vectorstore saved at {DB_FAISS_PATH}")
