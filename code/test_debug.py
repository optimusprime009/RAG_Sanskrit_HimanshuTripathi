import sys
import os

print("--- Python Executable ---")
print(sys.executable)

print("\n--- System Paths ---")
for p in sys.path:
    print(p)

print("\n--- Attempting Import ---")
try:
    import langchain
    print(f"LangChain Location: {langchain.__file__}")
    from langchain.chains import RetrievalQA
    print("SUCCESS: RetrievalQA found!")
except ImportError as e:
    print(f"FAILURE: {e}")
except Exception as e:
    print(f"ERROR: {e}")