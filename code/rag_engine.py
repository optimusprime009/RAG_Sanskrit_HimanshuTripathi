import sys
import os
import config
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate

# --- MANUAL RAG CLASS ---
class SimpleRAG:
    def __init__(self, llm, retriever, prompt):
        self.llm = llm
        self.retriever = retriever
        self.prompt = prompt

    def invoke(self, inputs):
        query = inputs["query"]
        docs = self.retriever.invoke(query)
        context_text = "\n\n".join([doc.page_content for doc in docs])
        final_prompt = self.prompt.format(context=context_text, question=query)
        answer = self.llm.invoke(final_prompt)
        return {
            "result": answer,
            "source_documents": docs
        }

def initialize_rag_chain():
    # 1. Verify Model
    if not os.path.exists(config.MODEL_PATH):
        print(f"CRITICAL ERROR: Model not found at {config.MODEL_PATH}")
        sys.exit(1)

    # 2. Load Vector DB
    print("Loading Vector Database...")
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )
    
    if not os.path.exists(config.VECTOR_DB_DIR):
        print("Vector DB not found. Please run ingest.py first.")
        sys.exit(1)
        
    vector_db = Chroma(
        persist_directory=config.VECTOR_DB_DIR, 
        embedding_function=embeddings
    )

    # 3. Load LLM (CPU Mode)
    print("Initializing LLM (CPU)...")
    try:
        llm = LlamaCpp(
            model_path=config.MODEL_PATH,
            n_ctx=config.MODEL_N_CTX,
            n_batch=config.MODEL_N_BATCH,
            temperature=0.1, # Keep strict
            n_gpu_layers=0,
            verbose=False
        )
    except Exception as e:
        print(f"Error loading LLM: {e}")
        sys.exit(1)

    # 4. UPDATED PROMPT (The Fix)
    # We explicitly tell it to translate the relevant Sanskrit parts.
    prompt_template = """System: You are a Sanskrit scholar. Read the Sanskrit context below and answer the question in English.

    Context:
    {context}

    User Question: {question}

    Instructions:
    1. Find the answer in the Sanskrit text.
    2. Translate the relevant sentence to English.
    3. If the servant brought something in a 'torn cloth' (Jirne Vastre), mention that.
    4. If you don't find the answer, say "I don't know".

    Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )

    # 5. Create Chain
    qa_chain = SimpleRAG(
        llm=llm, 
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}), 
        prompt=PROMPT
    )

    return qa_chain