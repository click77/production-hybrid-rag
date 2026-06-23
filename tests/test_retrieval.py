# tests/test_retrieval.py
from src.retrieval.dense import dense_search
from src.retrieval.sparse import sparse_search

query = "how does hybrid retrieval work"

print("\n══ DENSE RETRIEVAL ══")
dense_results = dense_search(query, top_k=3)
for r in dense_results:
    print(f"  similarity={r['similarity']} | {r['text'][:100]}...")

print("\n══ SPARSE RETRIEVAL ══")
sparse_results = sparse_search(query, top_k=3)
for r in sparse_results:
    print(f"  bm25={r['bm25_score']} | {r['text'][:100]}...")