# tests/test_scorer.py
from src.retrieval.reranker import retrieve_and_rerank
from src.generation.generator import generate_answer
from src.generation.citation_verifier import verify_citations
from src.generation.scorer import score_and_gate

# ── Test 1: normal query that should be answerable ──
print("\n" + "═"*50)
print("TEST 1 — answerable query")
print("═"*50)

query  = "how does hybrid retrieval work"
chunks = retrieve_and_rerank(query, fusion_top_k=20, rerank_top_k=5)
result = generate_answer(query, chunks)
verified = verify_citations(result)
final  = score_and_gate(query, verified, chunks)

print(f"\nIs refusal : {final.get('is_refusal')}")
print(f"Composite  : {final['scores']['composite']['score']:.0%}")
print(f"Quality    : {final['scores']['composite']['quality']}")
print(f"\nAnswer:\n{final.get('answer', '')[:400]}...")

# ── Test 2: unanswerable query to trigger graceful refusal ──
print("\n" + "═"*50)
print("TEST 2 — unanswerable query (should trigger refusal)")
print("═"*50)

query2  = "what is the capital of mars and its tax policy"
chunks2 = retrieve_and_rerank(query2, fusion_top_k=20, rerank_top_k=5)
result2 = generate_answer(query2, chunks2)
verified2 = verify_citations(result2)
final2  = score_and_gate(query2, verified2, chunks2)

print(f"\nIs refusal : {final2.get('is_refusal')}")
print(f"\nRefusal message:\n{final2.get('answer', '')}")