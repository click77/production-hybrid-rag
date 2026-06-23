# tests/test_chunker.py
from src.ingestion.loader import load_document
from src.ingestion.chunker import chunk_documents, ChunkStrategy

doc = load_document("docs/sample.txt")

print("\n── Strategy 1: Fixed ──")
chunks = chunk_documents(doc, strategy=ChunkStrategy.FIXED)
for c in chunks:
    print(f"  [{c['chunk_index']}] {c['token_count']} tokens | {c['text'][:80]}...")

print("\n── Strategy 2: Recursive ──")
chunks = chunk_documents(doc, strategy=ChunkStrategy.RECURSIVE)
for c in chunks:
    print(f"  [{c['chunk_index']}] heading={c['section_heading']} | {c['text'][:80]}...")

print("\n── Strategy 3: Semantic ──")
chunks = chunk_documents(doc, strategy=ChunkStrategy.SEMANTIC)
for c in chunks:
    print(f"  [{c['chunk_index']}] {c['token_count']} tokens | {c['text'][:80]}...")