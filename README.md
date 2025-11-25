# 🌺 Sanskrit Document Retrieval-Augmented Generation System

### **CPU-Optimized RAG for Classical Sanskrit Text Understanding**

---

## 🧭 1. Introduction

Understanding Sanskrit literature often requires domain knowledge, contextual understanding, and the ability to interpret classical grammar. This project solves that problem using a **Retrieval-Augmented Generation (RAG)** system that:

* Reads Sanskrit documents
* Converts them into meaningful vector embeddings
* Retrieves the most relevant parts
* Uses a lightweight LLM (**Phi-3 Mini Quantized**) to generate human-readable answers

This system is specifically optimized to run on **CPU-only machines**, making it accessible for any student, intern, or researcher without needing a GPU.

---

## 🏗️ 2. System Architecture

```
                    ┌─────────────────────────────────────────────────┐
                    │              Sanskrit Documents                 │
                    │  (docx, txt, pdf → processed into text chunks) │
                    └─────────────────────────────────────────────────┘
                                      │
                                      ▼
                      ┌─────────────────────────────────┐
                      │ Text Preprocessing & Chunking   │
                      │ (ingest.py)                     │
                      └─────────────────────────────────┘
                                      │
                                      ▼
                  ┌──────────────────────────────────────────┐
                  │   Embeddings Model (HuggingFace)         │
                  │   → Generates vector representation       │
                  └──────────────────────────────────────────┘
                                      │
                                      ▼
                  ┌──────────────────────────────────────────┐
                  │ Vector Store (ChromaDB)                  │
                  │ → Stores and indexes embedded chunks     │
                  └──────────────────────────────────────────┘
                                      │
                                      ▼
           ┌─────────────────────────────────────────────────────────┐
           │     RAG Engine (rag_engine.py)                          │
           │  1. Retrieve top-k relevant chunks                      │
           │  2. Pass them into LLM prompt                           │
           │  3. Generate grounded output                            │
           └─────────────────────────────────────────────────────────┘
                                      │
                                      ▼
              ┌────────────────────────────────────────────┐
              │      LLM (Phi-3 Mini Quantized GGUF)       │
              │ CPU inference via llama-cpp-python         │
              └────────────────────────────────────────────┘
                                      │
                                      ▼
                         📝 Final Answer to User
```

---

## ⚡ 3. Key Features

### ✔ CPU-Optimized

Runs on an ordinary laptop with no GPU.

### ✔ Works with Sanskrit

Supports Sanskrit → English understanding.

### ✔ Fully Modular

Components separated into configuration, ingestion, retrieval, and generation.

### ✔ Lightweight LLM (Phi-3 Mini)

Fast, accurate, and quantized for low memory usage.

### ✔ Vector Database Using ChromaDB

Efficient semantic search on large text documents.

### ✔ CLI-Based Query System

User can ask real-time questions about Sanskrit documents.

---

## 📂 4. Repository Structure (Expanded)

```text
RAG_Sanskrit_Himanshu_Tripathi/
│
├── code/
│   ├── config.py
│   │     ├─ Contains model paths, vector DB paths
│   │     ├─ Embedding model selection
│   │     └─ Chunk size and RAG parameters
│   │
│   ├── ingest.py
│   │     ├─ Loads Sanskrit docx
│   │     ├─ Cleans formatting
│   │     ├─ Splits text into chunks
│   │     ├─ Converts to embeddings
│   │     └─ Saves ChromaDB vector store
│   │
│   ├── rag_engine.py
│   │     ├─ Defines retriever
│   │     ├─ Defines LLM pipeline
│   │     └─ Full RAG chain logic
│   │
│   ├── main.py
│   │     ├─ Command-line interface
│   │     ├─ User input → RAG query
│   │     └─ Pretty printing of results
│   │
│   └── requirements.txt
│
├── data/
│   ├── source_docs/
│   │     └── Rag-docs.docx
│   └── vector_store/
│         └── (Auto-generated ChromaDB files)
│
├── models/
│   └── phi-3-mini-4k-instruct.Q4_K_M.gguf
│
├── report/
│   └── Technical_Report.pdf
│
└── README.md
```

---

## 🧪 5. Theory Behind the System

### 5.1 What is RAG?

**RAG = Retrieval + Generation**

Instead of relying on the LLM to “know” the answer, it retrieves relevant information from documents. This ensures:

* High accuracy
* Grounded answers
* No hallucination

---

### 5.2 Why Phi-3 Mini?

| Feature        | Value        |
| -------------- | ------------ |
| Params         | ~3.8B        |
| Context Length | 4K tokens    |
| Quantized Size | ~2.2GB       |
| Hardware       | CPU-friendly |
| License        | Permissive   |

This makes it a powerful yet lightweight model for classical texts.

---

### 5.3 Embedding Model Used

Uses **HuggingFace Transformers Sentence Embeddings** such as:

```
sentence-transformers/all-mpnet-base-v2
```

Works excellently on multilingual (including Sanskrit) text.

---

## ⚙️ 6. Installation Guide

### Step 1: Navigate to Folder

```bash
cd RAG_Sanskrit_Himanshu_Tripathi
```

### Step 2: Install Requirements

```bash
pip install -r code/requirements.txt
```

### Step 3: Download Phi-3 Mini GGUF Model

1. Visit HuggingFace
2. Search: **"Phi-3-mini-4k-instruct-q4.gguf"**
3. Download
4. Rename to:

```
phi-3-mini-4k-instruct.Q4_K_M.gguf
```

5. Move file into:

```
/models/
```

---

## 🏃‍♂️ 7. Running the System

### Step 1: Build Vector Store

```bash
python code/ingest.py
```

You must see:

```
Success! Vector Database saved...
```

---

### Step 2: Run the Query Interface

```bash
python code/main.py
```

---

### Step 3: Interact

**User →**

```
What did the servant bring instead of sugar?
```

**RAG System →**
A context-grounded answer extracted from the Sanskrit text.

**User →**

```
Who is Kalidasa?
```

**User →**

```
exit
```

---

## 📊 8. Example Output (Illustrative)

```
User: Who is the king mentioned in the second paragraph?

Retrieved Context:
"...राजा पृथ्वीपालः ..."

Model Response:
The king referenced in the second section is **Prithvipala**, a ruler described
as just and devoted to dharma.
```

---

## 🛠️ 9. Troubleshooting

### ❌ Missing LangChain Modules

```bash
python -m pip install -U langchain langchain-community langchain-core
```

### ❌ Tokenizer Version Errors

```bash
pip install "tokenizers>=0.21,<0.22"
```

### ❌ ChromaDB Permission Issues

Delete and regenerate:

```bash
rm -rf data/vector_store/*
python code/ingest.py
```

---

## 💡 10. Future Enhancements

* Web interface (FastAPI / Streamlit)
* OCR support for Sanskrit PDFs
* GPU acceleration option
* Support for multiple documents
* Integration with cloud storage

---

## 👨‍💻 11. Intern Details

**Name:** *Himanshu Tripathi*
**Project:** AI/ML Internship — Sanskrit RAG System
**Institute:** Birla Institute of Technology, Noida (BIT)
**Date:** November 2025