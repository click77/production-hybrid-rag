# src/retrieval/fusion.py

import os
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# CONFIGURABLE WEIGHTS (from .env)
# ─────────────────────────────────────────────

DENSE_WEIGHT  = float(os.getenv("DENSE_WEIGHT",  0.7))
SPARSE_WEIGHT = float(os.getenv("SPARSE_WEIGHT", 0.3))
RRF_K         = 60   # standard RRF constant — dampens the impact of rank


# ─────────────────────────────────────────────
# RECIPROCAL RANK FUSION
# ─────────────────────────────────────────────

def reciprocal_rank_fusion(
    dense_results: list,
    sparse_results: list,
    top_k: int = 20,
    dense_weight: float = None,
    sparse_weight: float = None,
) -> list:
    """
    Merge dense (vector) and sparse (BM25) result lists into a single
    ranked list using Reciprocal Rank Fusion (RRF).

    WHY RRF:
    Raw similarity scores from different retrieval methods are not
    comparable — a BM25 score of 8.3 means something completely different
    to a cosine similarity of 0.82. RRF solves this by ignoring the raw
    scores entirely and working only with RANK POSITIONS.

    HOW IT WORKS:
    For each chunk that appears in either result list:
        rrf_score = dense_weight  * 1/(rank_in_dense  + K)
                  + sparse_weight * 1/(rank_in_sparse + K)

    K=60 is the standard constant — it prevents a rank-1 result from
    dominating too heavily, and makes results at rank 60+ contribute
    almost nothing.

    A chunk that ranks #1 in dense and #1 in sparse gets the highest
    possible score. A chunk only in one list gets a partial score.
    A chunk absent from a list gets 0 contribution from that side.

    Args:
        dense_results  — output of dense_search()
        sparse_results — output of sparse_search()
        top_k          — how many merged results to return (default 20)
        dense_weight   — override .env weight for this call
        sparse_weight  — override .env weight for this call

    Returns:
        List of chunk dicts sorted by rrf_score descending, each with:
            rrf_score, dense_rank, sparse_rank, dense_weight,
            sparse_weight, plus all original chunk fields
    """
    dw = dense_weight  if dense_weight  is not None else DENSE_WEIGHT
    sw = sparse_weight if sparse_weight is not None else SPARSE_WEIGHT

    # Normalise weights so they always sum to 1.0
    total = dw + sw
    dw    = dw / total
    sw    = sw / total

    print(f"\n  [Fusion] Merging {len(dense_results)} dense + "
          f"{len(sparse_results)} sparse results "
          f"(weights: dense={dw:.2f}, sparse={sw:.2f})")

    # ── Build score accumulator keyed by chunk_id ──────────────────────
    scores    = {}   # chunk_id → rrf_score
    chunk_map = {}   # chunk_id → chunk dict (for reconstructing output)
    rank_map  = {}   # chunk_id → {dense_rank, sparse_rank}

    # ── Score dense results ────────────────────────────────────────────
    for rank, chunk in enumerate(dense_results, start=1):
        cid = chunk["chunk_id"]
        rrf_contribution = dw * (1 / (rank + RRF_K))

        scores[cid]    = scores.get(cid, 0) + rrf_contribution
        chunk_map[cid] = chunk
        rank_map[cid]  = rank_map.get(cid, {})
        rank_map[cid]["dense_rank"] = rank

    # ── Score sparse results ───────────────────────────────────────────
    for rank, chunk in enumerate(sparse_results, start=1):
        cid = chunk["chunk_id"]
        rrf_contribution = sw * (1 / (rank + RRF_K))

        scores[cid]    = scores.get(cid, 0) + rrf_contribution
        chunk_map[cid] = chunk_map.get(cid, chunk)
        rank_map[cid]  = rank_map.get(cid, {})
        rank_map[cid]["sparse_rank"] = rank

    # ── Sort by RRF score descending ───────────────────────────────────
    sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

    # ── Build output list ──────────────────────────────────────────────
    fused = []
    for final_rank, cid in enumerate(sorted_ids[:top_k], start=1):
        chunk = dict(chunk_map[cid])         # copy to avoid mutating original

        # Strip retrieval-method-specific score fields
        chunk.pop("similarity",     None)
        chunk.pop("bm25_score",     None)
        chunk.pop("bm25_score_raw", None)

        chunk.update({
            "rrf_score":      round(scores[cid], 6),
            "final_rank":     final_rank,
            "dense_rank":     rank_map[cid].get("dense_rank",  None),
            "sparse_rank":    rank_map[cid].get("sparse_rank", None),
            "dense_weight":   round(dw, 3),
            "sparse_weight":  round(sw, 3),
            "retrieval_type": "hybrid_rrf",
        })
        fused.append(chunk)

    print(f"  [Fusion] Produced {len(fused)} fused results "
          f"(top rrf_score: {fused[0]['rrf_score'] if fused else 'n/a'})")

    return fused


# ─────────────────────────────────────────────
# FULL HYBRID SEARCH — single entry point
# ─────────────────────────────────────────────

def hybrid_search(
    query: str,
    top_k: int = 20,
    dense_weight: float = None,
    sparse_weight: float = None,
) -> list:
    """
    Run dense + sparse retrieval then fuse with RRF.
    This is the single function you call from the generation layer.

    Args:
        query         — the user's question
        top_k         — number of fused results to return before reranking
        dense_weight  — override .env weight (optional)
        sparse_weight — override .env weight (optional)

    Returns:
        Fused ranked list of chunk dicts ready for the reranker.
    """
    from src.retrieval.dense  import dense_search
    from src.retrieval.sparse import sparse_search

    # Retrieve top 20 from each method before fusion
    # More candidates = better fusion quality
    dense_results  = dense_search(query,  top_k=20)
    sparse_results = sparse_search(query, top_k=20)

    return reciprocal_rank_fusion(
        dense_results,
        sparse_results,
        top_k=top_k,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
    )