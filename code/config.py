import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# Data Paths
DATA_SOURCE_DIR = os.path.join(PROJECT_ROOT, "data", "source_docs")
VECTOR_DB_DIR = os.path.join(PROJECT_ROOT, "data", "vector_store")

# Model Path (Must match your downloaded file exactly)
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "phi-3-mini-4k-instruct.Q4_K_M.gguf")

# LLM CPU Settings
MODEL_N_CTX = 2048   # Context window
MODEL_N_BATCH = 512  # Batch processing size
MODEL_TEMP = 0.1     # Low temperature for factual answers

# Ingestion Settings
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"