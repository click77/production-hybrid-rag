# src/retrieval/reranker.py

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

RERANK_MODEL    = "gpt-4o-mini"   # fast + cheap for scoring
RERANK_TOP_K    = 5               # how many to keep after reranking
RERANK_BATCH    = 5               # how many chunks to score per API call


# ─────────────────────────────────────────────
# LLM-AS-JUDGE RERANKER
# ─────────────────────────────────────────────

def _score_batch(query: str, chunks: list) -> list:
    """
    Send a batch of chunks to GPT-4o-mini and ask it to score
    each one's relevance to the query on a scale of 0–10.

    Returns a list of floats (one score per chunk).

    Using a batch prompt rather than one call per chunk keeps
    costs low while still getting LLM-quality relevance judgments.
    """
    # Build numbered context blocks
    context_blocks = ""
    for i, chunk in enumerate(chunks, start=1):
        context_blocks += f"\n[{i}]\n{chunk['text'][:600]}\n"

    prompt = f"""You are a relevance scoring system for a RAG pipeline.

Given a user question and {len(chunks)} retrieved text chunks,
score each chunk's relevance to the question on a scale of 0 to 10:

  10 = directly and completely answers the question
   7 = highly relevant, contains key information
   4 = partially relevant, tangentially related
   1 = barely relevant
   0 = completely irrelevant

USER QUESTION:
{query}

RETRIEVED CHUNKS:
{context_blocks}

Respond with ONLY a valid JSON array of {len(chunks)} numbers.
Example for 3 chunks: [8, 3, 6]
No explanation. No markdown. Just the JSON array."""

    response = client.chat.completions.create(
        model=RERANK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,        # deterministic scoring
        max_tokens=100,
    )

    raw = response.choices[0].message.content.strip()

    try:
        scores = json.loads(raw)
        if isinstance(scores, list) and len(scores) == len(chunks):
            return [float(s) for s in scores]
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: return neutral scores if parsing fails
    print(f"  [Reranker] Warning: could not parse scores: {raw}")
    return [5.0] * len(chunks)


def rerank(
    query: str,
    chunks: list,
    top_k: int = RERANK_TOP_K,
) -> list:
    """
    Second-pass reranker: takes RRF output (up to 20 chunks) and
    uses GPT-4o-mini as a judge to score each chunk's true relevance
    to the query. Keeps only the top_k highest-scoring chunks.

    WHY THIS MATTERS:
    RRF merges rank positions but doesn't understand the actual question.
    A chunk can rank highly because it shares keywords (BM25) or
    embedding neighbourhood (dense) without actually answering the query.
    The reranker reads both the question AND the chunk text and makes
    a semantic judgment — dramatically improving precision.

    This is the step that separates a production RAG system from a demo.

    Args:
        query  — the user's original question
        chunks — output of hybrid_search() / reciprocal_rank_fusion()
        top_k  — how many chunks to keep (default 5)

    Returns:
        Top_k chunk dicts with rerank_score and rerank_rank added,
        sorted by rerank_score descending.
    """
    if not chunks:
        return []

    print(f"\n  [Reranker] Scoring {len(chunks)} candidates with {RERANK_MODEL}...")

    all_scores = []

    # Score in batches to keep prompt size manageable
    for i in range(0, len(chunks), RERANK_BATCH):
        batch  = chunks[i:i + RERANK_BATCH]
        scores = _score_batch(query, batch)
        all_scores.extend(scores)
        print(f"  [Reranker] Batch {i // RERANK_BATCH + 1} scored: {scores}")

    # Attach scores to chunks
    scored_chunks = []
    for chunk, score in zip(chunks, all_scores):
        scored = dict(chunk)
        scored["rerank_score"] = round(score, 2)
        scored_chunks.append(scored)

    # Sort by rerank score descending
    scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)

    # Keep top_k and add final rank
    final = []
    for rank, chunk in enumerate(scored_chunks[:top_k], start=1):
        chunk["rerank_rank"] = rank
        final.append(chunk)

    print(f"\n  [Reranker] Kept top {len(final)} chunks:")
    for c in final:
        print(f"    rank={c['rerank_rank']} "
              f"score={c['rerank_score']} "
              f"| {c['text'][:80]}...")

    return final


# ─────────────────────────────────────────────
# FULL PIPELINE — retrieve + fuse + rerank
# ─────────────────────────────────────────────

def retrieve_and_rerank(
    query: str,
    fusion_top_k: int = 20,
    rerank_top_k: int = 5,
    dense_weight: float = None,
    sparse_weight: float = None,
) -> list:
    """
    Full retrieval pipeline in one call:
        1. Dense search     → top 20
        2. Sparse search    → top 20
        3. RRF fusion       → top fusion_top_k (default 20)
        4. LLM reranker     → top rerank_top_k (default 5)

    This is what gets called by the generation layer.

    Returns:
        Final top_k chunks ready to be inserted into the generation prompt.
    """
    from src.retrieval.fusion import hybrid_search

    print(f"\n{'═'*50}")
    print(f"  RETRIEVAL PIPELINE")
    print(f"  Query: {query[:80]}")
    print(f"{'═'*50}")

    # Steps 1–3: hybrid search with RRF
    fused_chunks = hybrid_search(
        query,
        top_k=fusion_top_k,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
    )

    # Step 4: rerank
    final_chunks = rerank(query, fused_chunks, top_k=rerank_top_k)

    print(f"\n  Pipeline complete — {len(final_chunks)} chunks ready for generation")
    return final_chunks