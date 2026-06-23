# tests/test_generation.py
from src.retrieval.reranker import retrieve_and_rerank
from src.generation.generator import generate_answer, is_insufficient_context
from src.generation.citation_verifier import verify_citations

query = "how does hybrid retrieval work"

# Step 1 — retrieve and rerank
chunks = retrieve_and_rerank(query, fusion_top_k=20, rerank_top_k=5)

# Step 2 — generate grounded answer
result = generate_answer(query, chunks)

print("\n══ GENERATED ANSWER ══")
print(result["answer"])
print(f"\nCitations found: {result['citations_found']}")
print(f"Insufficient context: {is_insufficient_context(result['answer'])}")

# Step 3 — verify citations
verified = verify_citations(result)

print("\n══ VERIFICATION RESULTS ══")
print(f"Score:              {verified['verification_score']:.0%}")
print(f"Verified:           {verified['verified_citations']}")
print(f"Unsupported:        {verified['unsupported_citations']}")
print(f"Has hallucinations: {verified['has_hallucinations']}")

print("\n══ FLAGGED ANSWER ══")
print(verified["flagged_answer"])