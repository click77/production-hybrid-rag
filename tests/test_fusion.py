# tests/test_fusion.py
from src.retrieval.fusion import hybrid_search
from src.retrieval.reranker import retrieve_and_rerank

query = "how does hybrid retrieval work"

print("\n══ HYBRID SEARCH (RRF only) ══")
fused = hybrid_search(query, top_k=5)
for r in fused:
    print(f"  rank={r['final_rank']} "
          f"rrf={r['rrf_score']} "
          f"dense_rank={r['dense_rank']} "
          f"sparse_rank={r['sparse_rank']} "
          f"| {r['text'][:80]}...")

print("\n══ FULL PIPELINE (RRF + Reranker) ══")
final = retrieve_and_rerank(query, fusion_top_k=20, rerank_top_k=3)
for r in final:
    print(f"  rank={r['rerank_rank']} "
          f"rerank_score={r['rerank_score']} "
          f"| {r['text'][:80]}...")