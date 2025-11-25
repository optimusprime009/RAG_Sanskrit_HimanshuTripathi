import time
from rag_engine import initialize_rag_chain

def main():
    print("=========================================")
    print("   Sanskrit RAG System (CPU Only)")
    print("=========================================")
    
    # Initialize the RAG pipeline
    qa_chain = initialize_rag_chain()
    
    print("\nSystem Ready! Type 'exit' to stop.")
    
    while True:
        query = input("\n>> Enter Query (English/Sanskrit): ")
        
        if query.lower() in ['exit', 'quit']:
            break
            
        if not query.strip():
            continue
            
        print("Thinking...")
        start_time = time.time()
        
        try:
            # Execute Query
            response = qa_chain.invoke({"query": query})
            end_time = time.time()
            
            # Display Output
            print(f"\n--- Answer ({round(end_time - start_time, 2)}s) ---")
            print(response['result'])
            
            # Show Sources
            print("\n[Source Context]")
            for i, doc in enumerate(response['source_documents']):
                print(f"{i+1}. {doc.page_content[:100]}...")
                
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()