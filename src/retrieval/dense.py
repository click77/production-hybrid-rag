# src/retrieval/dense.py

import os
from dotenv import load_dotenv
from openai import OpenAI
from src.ingestion.indexer import get_chroma_collection, EMBED_MODEL, CHROMA_DIR
import chromadb
from chromadb.config import Settings

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def embed_query(query: str) -> list:
    """
    Embed a single user query using text-embedding-3-small.
    Same model used during indexing — must match or scores are meaningless.
    """
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=[query],
    )
    return response.data[0].embedding


# ─────────────────────────────────────────────
# DENSE RETRIEVAL
# ─────────────────────────────────────────────

def dense_search(
    query: str,
    top_k: int = 10,
    min_similarity: float = 0.0,
) -> list:
    """
    Retrieve the top-k most semantically similar chunks from ChromaDB.

    Steps:
    1. Embed the query with text-embedding-3-small
    2. Query ChromaDB using cosine similarity
    3. Convert distances to similarity scores
    4. Filter out results below min_similarity
    5. Return ranked list of chunk dicts with scores

    Args:
        query          — the user's question
        top_k          — how many results to return (default 10)
        min_similarity — filter out chunks below this score (0.0 = no filter)

    Returns:
        List of dicts sorted by similarity descending, each containing:
            chunk_id, text, similarity, source_file, page_number,
            section_heading, strategy, token_count
    """
    # Step 1 — embed the query
    print(f"\n  [Dense] Embedding query: '{query[:60]}...'")
    query_embedding = embed_query(query)

    # Step 2 — query ChromaDB
    collection = get_chroma_collection()

    if collection.count() == 0:
        print("  [Dense] Collection is empty — run indexer first.")
        return []

    # Clamp top_k to collection size
    top_k = min(top_k, collection.count())

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # Step 3 — convert distances to similarity scores
    # ChromaDB cosine distance = 1 - cosine_similarity
    chunks = []
    for i in range(len(results["ids"][0])):
        distance   = results["distances"][0][i]
        similarity = round(1 - distance, 4)

        # Step 4 — filter by minimum similarity
        if similarity < min_similarity:
            continue

        chunk = {
            "chunk_id":        results["ids"][0][i],
            "text":            results["documents"][0][i],
            "similarity":      similarity,
            "retrieval_type":  "dense",
            # unpack all stored metadata fields
            **results["metadatas"][0][i],
        }
        chunks.append(chunk)

    # Step 5 — sort by similarity descending (ChromaDB usually does this
    # already but we enforce it explicitly)
    chunks.sort(key=lambda x: x["similarity"], reverse=True)

    print(f"  [Dense] Returned {len(chunks)} results "
          f"(top similarity: {chunks[0]['similarity'] if chunks else 'n/a'})")

    return chunks


def dense_search_with_filter(
    query: str,
    top_k: int = 10,
    filter_by: dict = None,
) -> list:
    """
    Dense search with optional metadata filtering.
    Use this to restrict results to a specific file, file type, or page.

    Example:
        dense_search_with_filter(
            "what is BM25?",
            filter_by={"file_type": "pdf"}
        )

    ChromaDB where clause supports: $eq, $ne, $gt, $gte, $lt, $lte, $in
    """
    query_embedding = embed_query(query)
    collection      = get_chroma_collection()

    if collection.count() == 0:
        return []

    top_k = min(top_k, collection.count())

    kwargs = {
        "query_embeddings": [query_embedding],
        "n_results":        top_k,
        "include":          ["documents", "metadatas", "distances"],
    }

    if filter_by:
        # Build ChromaDB where clause from simple key-value pairs
        if len(filter_by) == 1:
            key, val = next(iter(filter_by.items()))
            kwargs["where"] = {key: {"$eq": val}}
        else:
            kwargs["where"] = {
                "$and": [{k: {"$eq": v}} for k, v in filter_by.items()]
            }

    results = collection.query(**kwargs)

    chunks = []
    for i in range(len(results["ids"][0])):
        similarity = round(1 - results["distances"][0][i], 4)
        chunks.append({
            "chunk_id":       results["ids"][0][i],
            "text":           results["documents"][0][i],
            "similarity":     similarity,
            "retrieval_type": "dense",
            **results["metadatas"][0][i],
        })

    chunks.sort(key=lambda x: x["similarity"], reverse=True)
    return chunks