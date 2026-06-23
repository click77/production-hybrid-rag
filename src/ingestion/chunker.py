# src/ingestion/chunker.py

import os
import re
from enum import Enum
from typing import Union
from dotenv import load_dotenv

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ─────────────────────────────────────────────
# STRATEGY ENUM
# ─────────────────────────────────────────────

class ChunkStrategy(Enum):
    FIXED       = "fixed"       # Strategy 1: fixed-size with overlap
    RECURSIVE   = "recursive"   # Strategy 2: structure-aware recursive split
    SEMANTIC    = "semantic"    # Strategy 3: semantic topic boundary split


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _count_tokens(text: str) -> int:
    """Approximate token count (4 chars ≈ 1 token)."""
    return len(text) // 4


def _make_chunk(
    text: str,
    doc: dict,
    chunk_index: int,
    strategy: ChunkStrategy,
    section_heading: str = None,
) -> dict:
    """
    Build a standardised chunk dict.
    Inherits metadata from the parent document.
    """
    return {
        "chunk_id":        f"{doc['doc_id']}_c{chunk_index}",
        "doc_id":          doc["doc_id"],
        "source_file":     doc["source_file"],
        "file_type":       doc["file_type"],
        "page_number":     doc["page_number"],
        "section_heading": section_heading or doc.get("section_heading"),
        "strategy":        strategy.value,   # ← which strategy made this chunk
        "chunk_index":     chunk_index,
        "text":            text.strip(),
        "token_count":     _count_tokens(text),
        "char_count":      len(text),
    }


# ─────────────────────────────────────────────
# STRATEGY 1 — FIXED SIZE WITH OVERLAP
# ─────────────────────────────────────────────

def chunk_fixed(
    doc: dict,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list:
    """
    Split text into fixed-size chunks measured in characters,
    with a sliding overlap window so context isn't lost at boundaries.

    chunk_size    — target size of each chunk in tokens (approx)
    chunk_overlap — how many tokens to repeat between chunks
    """
    # Convert token targets to character targets (4 chars ≈ 1 token)
    char_size    = chunk_size * 4
    char_overlap = chunk_overlap * 4

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=char_size,
        chunk_overlap=char_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    raw_chunks = splitter.split_text(doc["text"])

    return [
        _make_chunk(text, doc, i, ChunkStrategy.FIXED)
        for i, text in enumerate(raw_chunks)
        if text.strip()
    ]


# ─────────────────────────────────────────────
# STRATEGY 2 — RECURSIVE / STRUCTURE-AWARE
# ─────────────────────────────────────────────

def chunk_recursive(
    doc: dict,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list:
    """
    Split text by respecting document structure first:
    tries to break on Markdown headers, then paragraphs,
    then sentences, then words — only splitting mid-sentence
    as a last resort.

    This preserves section context far better than fixed splitting
    for structured documents (docs, wikis, READMEs).
    """
    char_size    = chunk_size * 4
    char_overlap = chunk_overlap * 4

    # For Markdown/structured docs — split on headers first
    if doc.get("file_type") in {"md", "txt"}:
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#",   "h1"),
                ("##",  "h2"),
                ("###", "h3"),
            ],
            strip_headers=False,
        )
        try:
            header_chunks = header_splitter.split_text(doc["text"])
            if header_chunks:
                results = []
                for hchunk in header_chunks:
                    heading = (
                        hchunk.metadata.get("h1")
                        or hchunk.metadata.get("h2")
                        or hchunk.metadata.get("h3")
                    )
                    # Sub-split any section that's still too large
                    sub_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=char_size,
                        chunk_overlap=char_overlap,
                        separators=["\n\n", "\n", ". ", " ", ""],
                    )
                    sub_chunks = sub_splitter.split_text(hchunk.page_content)
                    for text in sub_chunks:
                        if text.strip():
                            results.append(
                                _make_chunk(
                                    text, doc,
                                    len(results),
                                    ChunkStrategy.RECURSIVE,
                                    section_heading=heading,
                                )
                            )
                if results:
                    return results
        except Exception:
            pass  # fall through to generic recursive split

    # Generic recursive split for HTML, PDF, plain text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=char_size,
        chunk_overlap=char_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )
    raw_chunks = splitter.split_text(doc["text"])

    return [
        _make_chunk(text, doc, i, ChunkStrategy.RECURSIVE)
        for i, text in enumerate(raw_chunks)
        if text.strip()
    ]


# ─────────────────────────────────────────────
# STRATEGY 3 — SEMANTIC CHUNKING
# ─────────────────────────────────────────────

def _embed(text: str) -> list:
    """Get an embedding vector from OpenAI."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


def _cosine_similarity(a: list, b: list) -> float:
    """Cosine similarity between two vectors."""
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x ** 2 for x in a) ** 0.5
    mag_b = sum(x ** 2 for x in b) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def chunk_semantic(
    doc: dict,
    max_chunk_tokens: int = 512,
    similarity_threshold: float = 0.75,
) -> list:
    """
    Split text on topic boundaries detected via embedding similarity.

    How it works:
    1. Split the document into sentences
    2. Embed each sentence
    3. Compare consecutive sentence embeddings
    4. When similarity drops below the threshold → topic has changed → split here
    5. Merge sentences within a topic into one chunk
    6. If a merged chunk is too large, sub-split it with fixed strategy

    similarity_threshold — lower = more splits (more granular topics)
                           higher = fewer splits (broader topics)
    """
    # Split into sentences
    sentences = re.split(r"(?<=[.!?])\s+", doc["text"])
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) <= 1:
        return [_make_chunk(doc["text"], doc, 0, ChunkStrategy.SEMANTIC)]

    # Embed all sentences (batched to reduce API calls)
    print(f"    Embedding {len(sentences)} sentences for semantic chunking...")
    embeddings = []
    batch_size = 20
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch,
        )
        embeddings.extend([item.embedding for item in response.data])

    # Find topic boundaries
    boundaries = [0]  # always start a chunk at sentence 0
    for i in range(1, len(sentences)):
        sim = _cosine_similarity(embeddings[i - 1], embeddings[i])
        if sim < similarity_threshold:
            boundaries.append(i)   # topic shift detected here

    boundaries.append(len(sentences))  # end sentinel

    # Merge sentences between boundaries into chunks
    chunks = []
    max_chars = max_chunk_tokens * 4

    for b in range(len(boundaries) - 1):
        start = boundaries[b]
        end   = boundaries[b + 1]
        merged = " ".join(sentences[start:end])

        # If the merged chunk is too large, sub-split it
        if len(merged) > max_chars:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=max_chars,
                chunk_overlap=64 * 4,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            sub_chunks = splitter.split_text(merged)
            for text in sub_chunks:
                if text.strip():
                    chunks.append(
                        _make_chunk(text, doc, len(chunks), ChunkStrategy.SEMANTIC)
                    )
        else:
            if merged.strip():
                chunks.append(
                    _make_chunk(merged, doc, len(chunks), ChunkStrategy.SEMANTIC)
                )

    return chunks


# ─────────────────────────────────────────────
# PUBLIC API — single entry point
# ─────────────────────────────────────────────

def chunk_document(
    doc: dict,
    strategy: Union[ChunkStrategy, str] = ChunkStrategy.FIXED,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list:
    """
    Chunk a single document dict (from loader.py) using the chosen strategy.

    Args:
        doc           — a document dict from load_document()
        strategy      — ChunkStrategy enum or string: "fixed", "recursive", "semantic"
        chunk_size    — target chunk size in tokens
        chunk_overlap — overlap between chunks in tokens (not used for semantic)

    Returns:
        List of chunk dicts, each with full metadata including which strategy was used.
    """
    if isinstance(strategy, str):
        strategy = ChunkStrategy(strategy)

    if strategy == ChunkStrategy.FIXED:
        return chunk_fixed(doc, chunk_size, chunk_overlap)
    elif strategy == ChunkStrategy.RECURSIVE:
        return chunk_recursive(doc, chunk_size, chunk_overlap)
    elif strategy == ChunkStrategy.SEMANTIC:
        return chunk_semantic(doc, chunk_size)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def chunk_documents(
    docs: list,
    strategy: Union[ChunkStrategy, str] = ChunkStrategy.FIXED,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list:
    """
    Chunk a list of document dicts. Convenience wrapper over chunk_document().
    Returns a flat list of all chunks across all documents.
    """
    if isinstance(strategy, str):
        strategy = ChunkStrategy(strategy)

    all_chunks = []
    for doc in docs:
        chunks = chunk_document(doc, strategy, chunk_size, chunk_overlap)
        all_chunks.extend(chunks)
        print(f"  ✓ {doc['source_file']} → {len(chunks)} chunks ({strategy.value})")

    return all_chunks