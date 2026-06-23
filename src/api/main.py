# src/api/main.py

import os
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────

app = FastAPI(
    title="Production Hybrid RAG API",
    description="""
A production-grade RAG pipeline with hybrid retrieval (dense + sparse),
Reciprocal Rank Fusion, LLM reranking, grounded generation, citation
verification, and confidence scoring.

## Endpoints
- **POST /v1/ask** — ask a question, get a grounded cited answer
- **GET  /v1/documents** — list all indexed documents
- **POST /v1/ingest** — upload and index new documents

## Architecture
Dense retrieval (ChromaDB) + Sparse retrieval (BM25) → RRF Fusion →
LLM Reranker → GPT-4o Generation → Citation Verification → Confidence Scoring
    """,
    version="1.0.0",
    docs_url="/docs",       # Swagger UI at /docs
    redoc_url="/redoc",     # ReDoc at /redoc
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────────

class AskRequest(BaseModel):
    question:      str
    top_k:         int   = 5
    fusion_top_k:  int   = 20
    dense_weight:  float = 0.7
    sparse_weight: float = 0.3
    verify_citations: bool = True
    model:         str   = "gpt-4o"

    class Config:
        json_schema_extra = {
            "example": {
                "question":         "How does hybrid retrieval work?",
                "top_k":            5,
                "fusion_top_k":     20,
                "dense_weight":     0.7,
                "sparse_weight":    0.3,
                "verify_citations": True,
                "model":            "gpt-4o",
            }
        }


class ChunkResult(BaseModel):
    chunk_id:        str
    text:            str
    source_file:     str
    page_number:     int
    section_heading: Optional[str]
    strategy:        str
    rerank_score:    Optional[float]
    rrf_score:       Optional[float]
    retrieval_type:  Optional[str]


class ScoreDimension(BaseModel):
    score:     float
    reasoning: str


class Scores(BaseModel):
    retrieval_confidence: ScoreDimension
    citation_coverage:    ScoreDimension
    answer_completeness:  ScoreDimension
    composite:            dict


class AskResponse(BaseModel):
    question:              str
    answer:                str
    flagged_answer:        Optional[str]
    citations_found:       list
    verified_citations:    list
    unsupported_citations: list
    has_hallucinations:    bool
    is_refusal:            bool
    scores:                Optional[dict]
    chunks:                list
    model:                 str
    chunks_used:           int
    timestamp:             str


class DocumentInfo(BaseModel):
    doc_id:        str
    source_file:   str
    file_type:     str
    page_number:   int
    section_heading: Optional[str]
    strategy:      str
    token_count:   int


class DocumentsResponse(BaseModel):
    total_chunks:   int
    documents:      list
    unique_sources: list


class IngestResponse(BaseModel):
    filename:       str
    file_type:      str
    chunks_created: int
    chunks_indexed: int
    duplicates_skipped: int
    status:         str


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    """Health check — returns API status."""
    return {
        "status":  "online",
        "service": "Production Hybrid RAG API",
        "version": "1.0.0",
        "docs":    "/docs",
    }


@app.get("/health", tags=["Health"])
def health():
    """Detailed health check including index status."""
    try:
        from src.ingestion.indexer import get_chroma_collection
        collection = get_chroma_collection()
        chunk_count = collection.count()
        index_status = "ready"
    except Exception as e:
        chunk_count  = 0
        index_status = f"error: {e}"

    return {
        "status":       "healthy",
        "index_status": index_status,
        "chunk_count":  chunk_count,
        "timestamp":    datetime.utcnow().isoformat(),
    }


@app.post("/v1/ask", response_model=AskResponse, tags=["RAG"])
def ask(request: AskRequest):
    """
    Ask a question and receive a grounded, cited answer.

    **Pipeline:**
    1. Dense retrieval (ChromaDB cosine similarity)
    2. Sparse retrieval (BM25 keyword search)
    3. Reciprocal Rank Fusion merge
    4. LLM reranker (GPT-4o-mini as judge)
    5. Grounded generation (GPT-4o with citations)
    6. Citation verification (LLM-as-judge per claim)
    7. Confidence scoring (retrieval + citation + completeness)

    **Returns:**
    - Grounded answer with bracketed citations
    - Flagged answer marking any unsupported citations
    - Per-dimension confidence scores
    - Source chunks with metadata
    - Whether the answer is a graceful refusal
    """
    try:
        from src.retrieval.reranker            import retrieve_and_rerank
        from src.generation.generator          import generate_answer
        from src.generation.citation_verifier  import verify_citations
        from src.generation.scorer             import score_and_gate

        # Step 1–4: retrieve and rerank
        chunks = retrieve_and_rerank(
            query=request.question,
            fusion_top_k=request.fusion_top_k,
            rerank_top_k=request.top_k,
            dense_weight=request.dense_weight,
            sparse_weight=request.sparse_weight,
        )

        # Step 5: generate
        generation = generate_answer(
            query=request.question,
            chunks=chunks,
            model=request.model,
        )

        # Step 6: verify citations
        if request.verify_citations:
            verified = verify_citations(generation)
        else:
            verified = {
                **generation,
                "verification_results":  [],
                "verified_citations":    generation.get("citations_found", []),
                "unsupported_citations": [],
                "verification_score":    1.0,
                "has_hallucinations":    False,
                "flagged_answer":        generation.get("answer", ""),
            }

        # Step 7: score and gate
        final = score_and_gate(request.question, verified, chunks)

        # Serialise chunks for response
        chunk_dicts = [
            {
                "chunk_id":        c.get("chunk_id", ""),
                "text":            c.get("text", ""),
                "source_file":     c.get("source_file", ""),
                "page_number":     c.get("page_number", 1),
                "section_heading": c.get("section_heading"),
                "strategy":        c.get("strategy", ""),
                "rerank_score":    c.get("rerank_score"),
                "rrf_score":       c.get("rrf_score"),
                "retrieval_type":  c.get("retrieval_type"),
            }
            for c in chunks
        ]

        return AskResponse(
            question=             request.question,
            answer=               final.get("answer", ""),
            flagged_answer=       final.get("flagged_answer", ""),
            citations_found=      final.get("citations_found", []),
            verified_citations=   final.get("verified_citations", []),
            unsupported_citations=final.get("unsupported_citations", []),
            has_hallucinations=   final.get("has_hallucinations", False),
            is_refusal=           final.get("is_refusal", False),
            scores=               final.get("scores"),
            chunks=               chunk_dicts,
            model=                request.model,
            chunks_used=          len(chunks),
            timestamp=            datetime.utcnow().isoformat(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/documents", response_model=DocumentsResponse, tags=["Documents"])
def list_documents(
    file_type: Optional[str] = Query(None, description="Filter by type: pdf, txt, md, html"),
    limit:     int           = Query(100,  description="Max chunks to return"),
):
    """
    List all documents currently indexed in the vector store.

    Returns chunk-level metadata including source file, page number,
    section heading, chunking strategy, and token count.
    Useful for verifying what has been indexed before querying.
    """
    try:
        from src.ingestion.indexer import get_chroma_collection

        collection = get_chroma_collection()
        total      = collection.count()

        if total == 0:
            return DocumentsResponse(
                total_chunks=0,
                documents=[],
                unique_sources=[],
            )

        # Fetch up to limit chunks
        results = collection.get(
            limit=min(limit, total),
            include=["metadatas"],
        )

        docs = []
        seen_sources = set()

        for meta in results["metadatas"]:
            if file_type and meta.get("file_type") != file_type:
                continue

            docs.append({
                "doc_id":          meta.get("doc_id", ""),
                "source_file":     meta.get("source_file", ""),
                "file_type":       meta.get("file_type", ""),
                "page_number":     meta.get("page_number", 1),
                "section_heading": meta.get("section_heading", ""),
                "strategy":        meta.get("strategy", ""),
                "token_count":     meta.get("token_count", 0),
            })
            seen_sources.add(meta.get("source_file", ""))

        return DocumentsResponse(
            total_chunks=   total,
            documents=      docs,
            unique_sources= sorted(seen_sources),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/ingest", response_model=IngestResponse, tags=["Documents"])
def ingest_document(
    file:     UploadFile = File(...),
    strategy: str        = Query("recursive",
                                 description="Chunking strategy: fixed, recursive, semantic"),
):
    """
    Upload and index a new document at runtime.

    **Supported formats:** .txt, .md, .html, .pdf

    **Flow:**
    1. Save uploaded file to docs/
    2. Load and normalise text
    3. Chunk with chosen strategy
    4. Deduplicate against existing index
    5. Insert new chunks into ChromaDB + BM25

    The document is immediately available for querying after ingestion.
    """
    supported = {".txt", ".md", ".html", ".htm", ".pdf"}
    suffix    = Path(file.filename).suffix.lower()

    if suffix not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Supported: {supported}",
        )

    if strategy not in {"fixed", "recursive", "semantic"}:
        raise HTTPException(
            status_code=400,
            detail="strategy must be one of: fixed, recursive, semantic",
        )

    try:
        from src.ingestion.loader  import load_document
        from src.ingestion.chunker import chunk_documents, ChunkStrategy
        from src.ingestion.indexer import DualIndex

        # Save to docs/ so it persists
        dest = Path("docs") / file.filename
        dest.parent.mkdir(parents=True, exist_ok=True)

        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Load, chunk, index
        docs   = load_document(dest)
        chunks = chunk_documents(
            docs,
            strategy=ChunkStrategy(strategy),
        )

        index   = DualIndex()
        summary = index.add_chunks(chunks)

        return IngestResponse(
            filename=          file.filename,
            file_type=         suffix.lstrip("."),
            chunks_created=    len(chunks),
            chunks_indexed=    summary["inserted"],
            duplicates_skipped=summary["skipped"],
            status=            "indexed",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))