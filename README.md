# 🌺 Sanskrit Document Retrieval-Augmented Generation System  
### **CPU-Optimized RAG for Classical Sanskrit Text Understanding**  
Fully Dockerized • CPU-Friendly • Supports Sanskrit + English Queries

---

## 🧭 1. Introduction

Understanding Sanskrit literature requires contextual comprehension, classical grammar knowledge, and careful interpretation. This project solves that challenge using a **Retrieval-Augmented Generation (RAG)** pipeline that:

- Reads Sanskrit documents  
- Converts them into semantic embeddings  
- Retrieves the most relevant text chunk  
- Generates accurate contextual answers using a lightweight LLM  

This system is fully optimized for **CPU-only inference**, allowing anyone to run it without needing a GPU.

---

## 🏗️ 2. System Architecture

```

```
            ┌─────────────────────────────────────────────────┐
            │              Sanskrit Documents                 │
            │  (docx, txt, pdf → processed into text chunks)  │
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
          │   → Generates vector representation      │
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
   │  3. Generate grounded answer                            │
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

````

---

## ⚡ 3. Key Features

### ✔ CPU-Optimized  
Runs smoothly on any modern CPU—no GPU needed.

### ✔ Sanskrit-Supported  
Understands queries in **English and Sanskrit**, retrieves from Sanskrit documents.

### ✔ Dockerized Environment  
Run without installing Python or dependencies.

### ✔ Modular Code Structure  
Separate modules for ingestion, retrieval, configuration, and inference.

### ✔ Lightweight Local LLM (Phi-3 Mini)  
Fast, small, accurate model for classical literature tasks.

### ✔ Persistent Vector Store (ChromaDB)  
Efficient semantic retrieval for large Sanskrit documents.

---

## 📂 4. Repository Structure (Expanded)

```text
RAG_Sanskrit_Himanshu_Tripathi/
│
├── code/
│   ├── config.py              # Model paths, DB paths, constants
│   ├── ingest.py              # Build vector DB from Sanskrit documents
│   ├── rag_engine.py          # Retrieval + Generation pipeline
│   ├── main.py                # CLI interface
│   └── requirements.txt       # Python dependencies
│
├── data/
│   ├── source_docs/
│   │     └── Rag-docs.docx    # Your Sanskrit text
│   └── vector_store/          # Auto-generated ChromaDB files
│
├── models/
│   └── phi-3-mini-4k-instruct.Q4_K_M.gguf   # LLM (manual download)
│
├── report/
│   └── Technical_Report.pdf
│
├── Dockerfile                 # Container definition
├── docker-compose.yml         # Container orchestration
└── README.md
````

---

## 🧪 5. Theory Behind the System

### 5.1 What is RAG?

RAG = **Retrieval + Generation**

Instead of relying on the LLM to “know everything,” the system retrieves relevant parts of the document and then generates an answer *grounded* in those retrieved chunks.
This prevents hallucinations and keeps answers accurate.

---

### 5.2 Why Phi-3 Mini?

| Property       | Value        |
| -------------- | ------------ |
| Params         | ~3.8B        |
| Context Window | 4K tokens    |
| Quantized Size | ~2.2GB       |
| Hardware       | CPU-Friendly |

This makes it ideal for embedded/offline Sanskrit NLP.

---

### 5.3 Embeddings

Uses:

```
sentence-transformers/all-mpnet-base-v2
```

Provides excellent multilingual performance, including Sanskrit.

---

## ⚙️ 6. Native Installation Guide (Python)

### Step 1 — Navigate to Folder

```bash
cd RAG_Sanskrit_Himanshu_Tripathi
```

### Step 2 — Install Requirements

```bash
pip install -r code/requirements.txt
```

### Step 3 — Download the Model (Manual Step)

Go to HuggingFace → Search:
**Phi-3-mini-4k-instruct-q4.gguf**

Rename file to:

```
phi-3-mini-4k-instruct.Q4_K_M.gguf
```

Place it inside:

```
/models/
```

### Step 4 — Build Vector Store

```bash
python code/ingest.py
```

### Step 5 — Run Application

```bash
python code/main.py
```

Then type your queries, for example:

```
How did the servant carry the sugar?
```

---

## 🐳 7. Docker Deployment (Recommended)

This setup allows you to run the entire system without installing Python locally.

### Step 1 — Build Docker Image

```bash
docker-compose build
```

### Step 2 — Create Vector Database

```bash
docker-compose run --rm rag-app python code/ingest.py
```

### Step 3 — Launch Interactive RAG Application

```bash
docker-compose run --rm rag-app
```

You will see the CLI:

```
>> Enter Query (English/Sanskrit):
```

---

## 📊 8. Example Output

```
>> Enter Query: How did the servant carry the sugar?

[response]:
The servant carried the sugar in a torn cloth. Because of the torn cloth,
the sugar leaked out along the road.

[Sanskrit Evidence]
"शर्कराम् जीर्णे वस्त्रे न्यस्यति च ।
 तस्मात् जीर्णवस्त्रात् मार्गे एव सर्वापि शर्करा स्त्रवति ।"
```

---

## 🛠️ 9. Troubleshooting

### Missing LangChain Modules:

```bash
python -m pip install -U langchain langchain-community langchain-core
```

### Tokenizer Version Error:

```bash
pip install "tokenizers>=0.21,<0.22"
```

### ChromaDB Permission Errors:

```bash
rm -rf data/vector_store/*
python code/ingest.py
```

---

## 👨‍💻 10. Intern Details

**Name:** *Himanshu Tripathi*
**Project:** AI/ML Internship — Sanskrit RAG System
**Institute:** Birla Institute of Technology, Noida
**Date:** November 2025

---