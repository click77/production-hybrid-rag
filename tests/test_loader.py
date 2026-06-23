# tests/test_loader.py
from src.ingestion.loader import load_document, load_directory, save_to_raw_store

# Test a single .txt file
docs = load_document("docs/sample.txt")
print(docs[0]["section_heading"])
print(docs[0]["text"][:200])

# Test a whole folder
all_docs = load_directory("docs/")
save_to_raw_store(all_docs, "data/raw")