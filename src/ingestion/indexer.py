# src/ingestion/indexer.py

import os
import json
import pickle
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

import numpy as np
import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

EMBED_MODEL          = "text-embedding-3-small"
CHROMA_DIR           = "data/chroma"
BM25_PATH            = "data/bm25_index.pkl"
COLLECTION_NAME      = "hybrid_rag"
DEDUP_THRESHOLD      = 0.95   # cosine similarity above this = duplicate


# ─────────────────────────────────────────────
# EMBEDDING
# ─────────────────────────────────────────────

def embed_texts(texts: list) -> list:
    """
    Embed a list of texts using text-embedding-3-small.
    Batches requests to stay within API limits (max 100 per call).
    Returns a list of embedding vectors (list of floats).
    """
    all_embeddings = []
    batch_size = 100

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=batch,
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
        print(f"  Embedded batch {i // batch_size + 1} "
              f"({len(batch)} texts)")

    return all_embeddings


# ─────────────────────────────────────────────
# CHROMA (VECTOR STORE)
# ─────────────────────────────────────────────

def get_chroma_collection():
    """
    Get or create the ChromaDB collection.
    Persists to disk at data/chroma/ so indexes survive restarts.
    """
    Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)

    chroma_client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False),
    )

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # use cosine similarity
    )

    return collection


def is_duplicate(embedding: list, collection, threshold: float = DEDUP_THRESHOLD) -> bool:
    """
    Check if a chunk is a near-duplicate of something already in ChromaDB.
    Queries the collection for the nearest neighbour and checks cosine similarity.
    Returns True if a duplicate is found (similarity > threshold).
    """
    if collection.count() == 0:
        return False

    results = collection.query(
        query_embeddings=[embedding],
        n_results=1,
        include=["distances"],
    )

    if not results["distances"] or not results["distances"][0]:
        return False

    # ChromaDB cosine distance = 1 - cosine_similarity
    # So similarity = 1 - distance
    distance   = results["distances"][0][0]
    similarity = 1 - distance

    return similarity > threshold


# ─────────────────────────────────────────────
# BM25 (SPARSE INDEX)
# ─────────────────────────────────────────────

def tokenize(text: str) -> list:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return text.lower().split()


class BM25Index:
    """
    Wrapper around rank_bm25 that keeps chunk metadata alongside
    the index so we can return full chunk dicts from search results.
    Persists to disk as a pickle file.
    """

    def __init__(self):
        self.chunks    = []     # list of chunk dicts
        self.corpus    = []     # list of tokenized texts
        self.bm25      = None   # BM25Okapi instance

    def add(self, chunk: dict):
        """Add a single chunk to the BM25 index."""
        self.chunks.append(chunk)
        self.corpus.append(tokenize(chunk["text"]))
        self.bm25 = BM25Okapi(self.corpus)   # rebuild (fast for small corpus)

    def search(self, query: str, top_k: int = 10) -> list:
        """
        Search the BM25 index.
        Returns top_k chunk dicts ranked by BM25 score.
        """
        if self.bm25 is None or not self.chunks:
            return []

        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)

        # Pair each chunk with its score, sort descending
        scored = sorted(
            zip(self.chunks, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return [
            {**chunk, "bm25_score": float(score)}
            for chunk, score in scored[:top_k]
            if score > 0
        ]

    def save(self, path: str = BM25_PATH):
        """Save the BM25 index to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "corpus": self.corpus}, f)
        print(f"  BM25 index saved to {path}")

    def load(self, path: str = BM25_PATH):
        """Load the BM25 index from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self.corpus = data["corpus"]
        self.bm25   = BM25Okapi(self.corpus)
        print(f"  BM25 index loaded ({len(self.chunks)} chunks)")
        return self


# ─────────────────────────────────────────────
# DUAL INDEX — ATOMIC INSERT
# ─────────────────────────────────────────────

class DualIndex:
    """
    Wraps ChromaDB + BM25 into a single interface.
    Every insert writes to BOTH indexes atomically so they stay in sync.
    Deduplication runs before every insert.
    """

    def __init__(self):
        self.collection = get_chroma_collection()
        self.bm25_index = BM25Index()

        # Load existing BM25 index from disk if it exists
        if Path(BM25_PATH).exists():
            self.bm25_index.load()

    def add_chunks(self, chunks: list) -> dict:
        """
        Embed and insert a list of chunk dicts into both indexes.

        For each chunk:
        1. Generate embedding
        2. Check for near-duplicates in ChromaDB
        3. If not a duplicate → insert into ChromaDB AND BM25
        4. If duplicate → skip and log it

        Returns a summary dict with counts.
        """
        if not chunks:
            return {"inserted": 0, "skipped": 0, "total": 0}

        print(f"\n  Embedding {len(chunks)} chunks...")
        texts      = [c["text"] for c in chunks]
        embeddings = embed_texts(texts)

        inserted = 0
        skipped  = 0

        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = chunk["chunk_id"]

            # ── Deduplication check ──────────────────────
            if is_duplicate(embedding, self.collection):
                print(f"  ⚠ Skipped duplicate: {chunk_id}")
                skipped += 1
                continue

            # ── Build metadata for ChromaDB ──────────────
            # ChromaDB metadata values must be str, int, float, or bool
            metadata = {
                "doc_id":          chunk["doc_id"],
                "source_file":     chunk["source_file"],
                "file_type":       chunk["file_type"],
                "page_number":     int(chunk["page_number"]),
                "section_heading": chunk.get("section_heading") or "",
                "strategy":        chunk["strategy"],
                "chunk_index":     int(chunk["chunk_index"]),
                "token_count":     int(chunk["token_count"]),
            }

            # ── Insert into ChromaDB ─────────────────────
            self.collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk["text"]],
                metadatas=[metadata],
            )

            # ── Insert into BM25 ────────────────────────
            self.bm25_index.add(chunk)

            inserted += 1
            print(f"  ✓ Indexed: {chunk_id}")

        # Save BM25 to disk after every batch
        self.bm25_index.save()

        summary = {
            "inserted": inserted,
            "skipped":  skipped,
            "total":    len(chunks),
        }
        print(f"\n  Done — {inserted} inserted, {skipped} duplicates skipped")
        return summary

    def vector_search(self, query: str, top_k: int = 10) -> list:
        """
        Search ChromaDB by semantic similarity.
        Returns top_k results with similarity scores.
        """
        query_embedding = embed_texts([query])[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        for i in range(len(results["ids"][0])):
            output.append({
                "chunk_id":        results["ids"][0][i],
                "text":            results["documents"][0][i],
                "similarity":      round(1 - results["distances"][0][i], 4),
                **results["metadatas"][0][i],
            })

        return output

    def keyword_search(self, query: str, top_k: int = 10) -> list:
        """Search the BM25 index by keyword."""
        return self.bm25_index.search(query, top_k)

    def count(self) -> int:
        """Return total number of chunks in the vector store."""
        return self.collection.count()

    def reset(self):
        """
        Wipe both indexes completely.
        Use this when re-indexing from scratch.
        """
        chroma_client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        chroma_client.delete_collection(COLLECTION_NAME)
        self.collection = get_chroma_collection()
        self.bm25_index = BM25Index()

        if Path(BM25_PATH).exists():
            Path(BM25_PATH).unlink()

        print("  Both indexes wiped and reset.")