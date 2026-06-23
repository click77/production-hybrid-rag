# src/retrieval/sparse.py

import pickle
from pathlib import Path
from src.ingestion.indexer import BM25_PATH


# ─────────────────────────────────────────────
# LOAD BM25 INDEX
# ─────────────────────────────────────────────

def load_bm25_index():
    """
    Load the BM25 index from disk.
    The index is saved by indexer.py at data/bm25_index.pkl.
    Raises FileNotFoundError if indexer hasn't been run yet.
    """
    if not Path(BM25_PATH).exists():
        raise FileNotFoundError(
            f"BM25 index not found at {BM25_PATH}. "
            "Run the indexer first: python tests/test_indexer.py"
        )

    from rank_bm25 import BM25Okapi

    with open(BM25_PATH, "rb") as f:
        data = pickle.load(f)

    chunks = data["chunks"]
    corpus = data["corpus"]
    bm25   = BM25Okapi(corpus)

    print(f"  [Sparse] BM25 index loaded ({len(chunks)} chunks)")
    return bm25, chunks


def tokenize(text: str) -> list:
    """
    Simple whitespace + lowercase tokenizer.
    Must match the tokenizer used during indexing in indexer.py.
    """
    return text.lower().split()


# ─────────────────────────────────────────────
# SPARSE RETRIEVAL
# ─────────────────────────────────────────────

def sparse_search(
    query: str,
    top_k: int = 10,
    min_score: float = 0.0,
) -> list:
    """
    Retrieve the top-k chunks by BM25 keyword score.

    BM25 excels at finding exact matches for:
    - Technical terms and function names (e.g. "chunk_recursive")
    - Config keys and error codes (e.g. "OPENAI_API_KEY", "404")
    - Acronyms (e.g. "BM25", "RAG", "RRF")
    - Proper nouns that embeddings might generalise over

    Steps:
    1. Load BM25 index from disk
    2. Tokenize the query
    3. Score every chunk in the corpus
    4. Normalise scores to 0–1 range
    5. Filter by min_score and return top_k

    Args:
        query     — the user's question
        top_k     — how many results to return (default 10)
        min_score — filter out chunks below this normalised score

    Returns:
        List of dicts sorted by bm25_score descending, each containing:
            chunk_id, text, bm25_score, bm25_score_raw, retrieval_type,
            source_file, page_number, section_heading, strategy
    """
    print(f"\n  [Sparse] Searching for: '{query[:60]}...'")

    # Step 1 — load index
    bm25, chunks = load_bm25_index()

    # Step 2 — tokenize query
    tokens = tokenize(query)
    if not tokens:
        print("  [Sparse] Empty query after tokenization.")
        return []

    # Step 3 — score all chunks
    raw_scores = bm25.get_scores(tokens)

    # Step 4 — normalise scores to 0–1
    max_score = max(raw_scores) if max(raw_scores) > 0 else 1.0
    norm_scores = [s / max_score for s in raw_scores]

    # Step 5 — pair, filter, sort, slice
    scored_chunks = []
    for chunk, raw_score, norm_score in zip(chunks, raw_scores, norm_scores):
        if norm_score < min_score:
            continue
        if raw_score <= 0:
            continue

        scored_chunks.append({
            **chunk,
            "bm25_score":     round(norm_score, 4),   # normalised 0–1
            "bm25_score_raw": round(float(raw_score), 4),
            "retrieval_type": "sparse",
        })

    scored_chunks.sort(key=lambda x: x["bm25_score"], reverse=True)
    top_results = scored_chunks[:top_k]

    print(f"  [Sparse] Returned {len(top_results)} results "
          f"(top score: {top_results[0]['bm25_score'] if top_results else 'n/a'})")

    return top_results


def sparse_search_multi(
    queries: list,
    top_k: int = 10,
) -> list:
    """
    Run multiple keyword queries and merge results by taking the max
    score per chunk across all queries.

    Useful for multi-part questions where different parts hit different
    keyword clusters. For example:
        ["BM25 retrieval", "sparse keyword search"]
    """
    bm25, chunks = load_bm25_index()
    max_score_per_chunk = {}

    for query in queries:
        tokens     = tokenize(query)
        raw_scores = bm25.get_scores(tokens)
        max_raw    = max(raw_scores) if max(raw_scores) > 0 else 1.0

        for chunk, raw_score in zip(chunks, raw_scores):
            cid       = chunk["chunk_id"]
            norm      = raw_score / max_raw
            current   = max_score_per_chunk.get(cid, {}).get("bm25_score", 0)
            if norm > current:
                max_score_per_chunk[cid] = {
                    **chunk,
                    "bm25_score":     round(norm, 4),
                    "bm25_score_raw": round(float(raw_score), 4),
                    "retrieval_type": "sparse_multi",
                }

    results = sorted(
        max_score_per_chunk.values(),
        key=lambda x: x["bm25_score"],
        reverse=True,
    )
    return results[:top_k]