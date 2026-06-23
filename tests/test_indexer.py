# tests/test_indexer.py
from src.ingestion.loader import load_document
from src.ingestion.chunker import chunk_documents, ChunkStrategy
from src.ingestion.indexer import DualIndex

# ── Load and chunk the sample doc ──
docs   = load_document("docs/sample.txt")
chunks = chunk_documents(docs, strategy=ChunkStrategy.FIXED)

# ── Build the dual index ──
index = DualIndex()
index.reset()   # start fresh each test run
summary = index.add_chunks(chunks)

print(f"\nIndex contains {index.count()} chunks")

# ── Test vector search ──
print("\n── Vector search: 'how does hybrid retrieval work' ──")
results = index.vector_search("how does hybrid retrieval work", top_k=3)
for r in results:
    print(f"  [{r['similarity']}] {r['text'][:100]}...")

# ── Test keyword search ──
print("\n── Keyword search: 'BM25 dense retrieval' ──")
results = index.keyword_search("BM25 dense retrieval", top_k=3)
for r in results:
    print(f"  [{round(r['bm25_score'], 3)}] {r['text'][:100]}...")