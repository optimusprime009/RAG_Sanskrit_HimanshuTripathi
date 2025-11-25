import os
import glob
import shutil
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import config

def ingest_data():
    print("========================================")
    print("   Ingestion Pipeline (Word Docs)       ")
    print("========================================")
    print("--- 1. Loading Sanskrit Documents ---")
    
    if not os.path.exists(config.DATA_SOURCE_DIR):
        print(f"Error: Directory {config.DATA_SOURCE_DIR} not found.")
        return

    # Look for .docx files specifically
    files = glob.glob(os.path.join(config.DATA_SOURCE_DIR, "*.docx"))
    documents = []
    
    if not files:
        print(f"No .docx files found in {config.DATA_SOURCE_DIR}")
        return

    for file_path in files:
        print(f"Loading: {os.path.basename(file_path)}")
        try:
            loader = Docx2txtLoader(file_path)
            documents.extend(loader.load())
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")

    print(f"Loaded {len(documents)} document(s).")

    print("--- 2. Splitting Text ---")
    # Custom separators for Sanskrit (including danda '||')
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["||", "|", "。", "\n\n", "\n", " ", ""],
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print("--- 3. Generating Embeddings (CPU) ---")
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )

    print("--- 4. Creating Vector Store ---")
    # Clear old database to ensure clean ingestion
    if os.path.exists(config.VECTOR_DB_DIR):
        shutil.rmtree(config.VECTOR_DB_DIR)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=config.VECTOR_DB_DIR
    )
    print(f"Success! Vector Database saved to: {config.VECTOR_DB_DIR}")

if __name__ == "__main__":
    ingest_data()