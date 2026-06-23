# src/generation/scorer.py

import os
import re
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SCORER_MODEL = "gpt-4o-mini"

# Thresholds — tune these based on your eval results
RETRIEVAL_CONFIDENCE_THRESHOLD = 0.35
CITATION_COVERAGE_THRESHOLD    = 0.70
COMPLETENESS_THRESHOLD         = 0.50
COMPOSITE_THRESHOLD            = 0.50   # below this → trigger "I don't know"


# ─────────────────────────────────────────────
# DIMENSION 1 — RETRIEVAL CONFIDENCE
# ─────────────────────────────────────────────

def score_retrieval_confidence(chunks: list) -> dict:
    """
    How relevant were the retrieved chunks to the query?

    Combines three signals:
    1. Rerank scores   — LLM-judge relevance scores (0–10) from reranker.py
    2. RRF scores      — fusion scores from hybrid search
    3. Coverage        — how many of the top-k slots had strong results

    Returns a score from 0.0 to 1.0.
    A low score means the retrieval system couldn't find
    relevant content — the answer is likely to be poor.
    """
    if not chunks:
        return {
            "score":      0.0,
            "reasoning":  "No chunks were retrieved.",
            "signals":    {},
        }

    # Signal 1: average rerank score (normalised from 0–10 to 0–1)
    rerank_scores = [
        c.get("rerank_score", 5.0) / 10.0
        for c in chunks
    ]
    avg_rerank = sum(rerank_scores) / len(rerank_scores)

    # Signal 2: average RRF score (already small floats, normalise to 0–1)
    rrf_scores = [c.get("rrf_score", 0.0) for c in chunks]
    max_rrf    = max(rrf_scores) if max(rrf_scores) > 0 else 1.0
    avg_rrf    = (sum(rrf_scores) / len(rrf_scores)) / max_rrf

    # Signal 3: coverage — proportion of chunks with rerank score >= 6/10
    strong_chunks = sum(1 for c in chunks if c.get("rerank_score", 0) >= 6.0)
    coverage      = strong_chunks / len(chunks)

    # Weighted composite
    score = (
        0.50 * avg_rerank +
        0.25 * avg_rrf    +
        0.25 * coverage
    )
    score = round(min(max(score, 0.0), 1.0), 4)

    return {
        "score":     score,
        "reasoning": (
            f"Avg rerank score {avg_rerank:.2f}, "
            f"RRF signal {avg_rrf:.2f}, "
            f"{strong_chunks}/{len(chunks)} chunks scored ≥6/10"
        ),
        "signals": {
            "avg_rerank_normalised": round(avg_rerank, 4),
            "avg_rrf_normalised":    round(avg_rrf, 4),
            "strong_chunk_coverage": round(coverage, 4),
            "chunks_evaluated":      len(chunks),
        },
    }


# ─────────────────────────────────────────────
# DIMENSION 2 — CITATION COVERAGE
# ─────────────────────────────────────────────

def score_citation_coverage(verified_result: dict) -> dict:
    """
    What proportion of claims in the answer have verified citations?

    Uses the output of citation_verifier.verify_citations().
    Penalises both missing citations AND unsupported citations.

    Score breakdown:
    - Starts at 1.0
    - Deducts for unsupported citations (harder penalty)
    - Deducts for uncited sentences (softer penalty)
    """
    answer = verified_result.get("answer", "")
    verification_results  = verified_result.get("verification_results",  [])
    unsupported_citations = verified_result.get("unsupported_citations", [])
    verified_citations    = verified_result.get("verified_citations",    [])

    # Count sentences that make claims (rough heuristic)
    sentences = [
        s.strip() for s in re.split(r"(?<=[.!?])\s+", answer)
        if s.strip() and len(s.strip()) > 20
    ]
    total_sentences = max(len(sentences), 1)

    # Count cited sentences
    cited_sentences = sum(
        1 for s in sentences
        if re.search(r"\[\d+\]", s)
    )

    # Metrics
    citation_rate = cited_sentences / total_sentences
    total_checked = len(verification_results)
    total_supported = len([r for r in verification_results if r["supported"]])
    support_rate  = total_supported / total_checked if total_checked > 0 else 1.0

    # Composite: weight citation rate + support rate
    score = round(0.40 * citation_rate + 0.60 * support_rate, 4)
    score = min(max(score, 0.0), 1.0)

    return {
        "score":     score,
        "reasoning": (
            f"{cited_sentences}/{total_sentences} sentences cited, "
            f"{total_supported}/{total_checked} citations verified"
        ),
        "signals": {
            "total_sentences":     total_sentences,
            "cited_sentences":     cited_sentences,
            "citation_rate":       round(citation_rate, 4),
            "citations_verified":  total_supported,
            "citations_checked":   total_checked,
            "support_rate":        round(support_rate, 4),
            "unsupported_list":    unsupported_citations,
            "verified_list":       verified_citations,
        },
    }


# ─────────────────────────────────────────────
# DIMENSION 3 — ANSWER COMPLETENESS
# ─────────────────────────────────────────────

def score_answer_completeness(query: str, answer: str) -> dict:
    """
    Did the answer address all parts of the question?

    Uses GPT-4o-mini as a judge to:
    1. Identify sub-questions or required elements in the query
    2. Check which elements are addressed in the answer
    3. Return a completeness score and breakdown

    This catches cases where the answer is factually grounded
    but only addresses part of what was asked.
    """
    prompt = f"""You are an answer completeness evaluator for a RAG system.

Given a QUESTION and an ANSWER, assess how completely the answer
addresses all parts of the question.

QUESTION:
{query}

ANSWER:
{answer}

Steps:
1. List the distinct sub-questions or required elements in the question
2. For each element, state whether the answer addresses it (yes/partial/no)
3. Calculate a completeness score from 0.0 to 1.0

Respond with ONLY valid JSON in this exact format:
{{
  "elements": [
    {{"element": "what BM25 is",        "addressed": "yes"}},
    {{"element": "how BM25 is computed", "addressed": "partial"}},
    {{"element": "BM25 vs TF-IDF",      "addressed": "no"}}
  ],
  "score":     0.67,
  "reasoning": "2 of 3 elements addressed fully or partially"
}}"""

    response = client.chat.completions.create(
        model=SCORER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=400,
    )

    raw = response.choices[0].message.content.strip()

    try:
        result  = json.loads(raw)
        score   = float(result.get("score", 0.5))
        score   = round(min(max(score, 0.0), 1.0), 4)

        return {
            "score":     score,
            "reasoning": result.get("reasoning", ""),
            "signals": {
                "elements": result.get("elements", []),
            },
        }
    except (json.JSONDecodeError, ValueError):
        print(f"  [Scorer] Warning: could not parse completeness: {raw}")
        return {
            "score":     0.5,
            "reasoning": "Could not parse completeness response.",
            "signals":   {"elements": []},
        }


# ─────────────────────────────────────────────
# COMPOSITE SCORER
# ─────────────────────────────────────────────

def score_answer(
    query: str,
    verified_result: dict,
    chunks: list,
) -> dict:
    """
    Score a generated answer across all three dimensions and
    return a composite score.

    Dimension weights:
        retrieval_confidence  — 0.35  (garbage in = garbage out)
        citation_coverage     — 0.40  (groundedness is paramount)
        answer_completeness   — 0.25  (did we answer the whole question)

    Args:
        query           — original user question
        verified_result — output of citation_verifier.verify_citations()
        chunks          — reranked chunks passed to the generator

    Returns a full scoring report dict attached to the answer.
    """
    print(f"\n  [Scorer] Scoring answer across 3 dimensions...")

    # Score each dimension
    retrieval  = score_retrieval_confidence(chunks)
    citation   = score_citation_coverage(verified_result)
    completeness = score_answer_completeness(
        query,
        verified_result.get("answer", "")
    )

    print(f"  [Scorer] Retrieval confidence : {retrieval['score']:.2f}")
    print(f"  [Scorer] Citation coverage    : {citation['score']:.2f}")
    print(f"  [Scorer] Answer completeness  : {completeness['score']:.2f}")

    # Weighted composite
    composite = round(
        0.35 * retrieval["score"]    +
        0.40 * citation["score"]     +
        0.25 * completeness["score"],
        4,
    )

    # Determine quality tier
    if composite >= 0.80:
        quality = "high"
    elif composite >= 0.55:
        quality = "medium"
    else:
        quality = "low"

    below_threshold = composite < COMPOSITE_THRESHOLD

    print(f"  [Scorer] Composite score      : {composite:.2f} ({quality})")
    if below_threshold:
        print(f"  [Scorer] ⚠ Below threshold — will trigger graceful refusal")

    scores = {
        "retrieval_confidence": retrieval,
        "citation_coverage":    citation,
        "answer_completeness":  completeness,
        "composite": {
            "score":           composite,
            "quality":         quality,
            "below_threshold": below_threshold,
            "weights": {
                "retrieval_confidence": 0.35,
                "citation_coverage":    0.40,
                "answer_completeness":  0.25,
            },
        },
    }

    return {**verified_result, "scores": scores}


# ─────────────────────────────────────────────
# GRACEFUL REFUSAL — "I DON'T KNOW" HANDLER
# ─────────────────────────────────────────────

def build_graceful_refusal(
    query: str,
    chunks: list,
    scores: dict,
) -> dict:
    """
    When retrieval confidence is too low to generate a trustworthy
    answer, return a structured refusal instead of hallucinating.

    The refusal includes:
    1. A clear statement that the system can't answer confidently
    2. What relevant content WAS found (if anything)
    3. Which source documents might be worth checking manually
    4. The confidence scores so the user understands why

    A well-designed refusal is more useful than a hallucinated answer
    and signals production maturity to anyone reviewing the system.
    """
    retrieval_score = scores.get(
        "retrieval_confidence", {}
    ).get("score", 0.0)

    # Gather what was found (chunks with any relevance signal)
    found_sources = []
    for chunk in chunks:
        source  = chunk.get("source_file", "unknown")
        heading = chunk.get("section_heading") or "—"
        page    = chunk.get("page_number", 1)
        rscore  = chunk.get("rerank_score", 0)

        if rscore >= 3.0:   # some relevance
            found_sources.append({
                "source":  source,
                "heading": heading,
                "page":    page,
                "score":   rscore,
            })

    # Deduplicate sources by file
    seen      = set()
    unique_sources = []
    for s in found_sources:
        if s["source"] not in seen:
            seen.add(s["source"])
            unique_sources.append(s)

    # Build refusal message
    if not chunks or retrieval_score < 0.15:
        found_text = "No relevant content was found in the document corpus."
    else:
        found_lines = "\n".join(
            f"  - {s['source']} (page {s['page']}, section: {s['heading']})"
            for s in unique_sources[:5]
        )
        found_text = (
            f"The following documents contained marginally related content "
            f"but not enough to answer confidently:\n{found_lines}"
        )

    refusal_message = f"""I was unable to find sufficient information in the \
document corpus to answer this question confidently.

QUESTION ASKED:
{query}

WHAT WAS FOUND:
{found_text}

SUGGESTED NEXT STEPS:
- Check whether the relevant documents have been indexed
- Try rephrasing the question using different keywords
- Review the source documents listed above manually

CONFIDENCE SCORES:
  Retrieval confidence : {scores.get('retrieval_confidence', {}).get('score', 0):.0%}
  Citation coverage    : {scores.get('citation_coverage',    {}).get('score', 0):.0%}
  Answer completeness  : {scores.get('answer_completeness',  {}).get('score', 0):.0%}
  Composite            : {scores.get('composite',            {}).get('score', 0):.0%}"""

    return {
        "answer":          refusal_message,
        "query":           query,
        "is_refusal":      True,
        "retrieval_score": retrieval_score,
        "sources_found":   unique_sources,
        "scores":          scores,
    }


# ─────────────────────────────────────────────
# FULL PIPELINE WITH REFUSAL GATE
# ─────────────────────────────────────────────

def score_and_gate(
    query: str,
    verified_result: dict,
    chunks: list,
) -> dict:
    """
    Score the answer and decide whether to return it or refuse.

    Flow:
        1. Score all three dimensions + composite
        2. If composite >= threshold → return scored answer
        3. If composite <  threshold → return graceful refusal

    This is the final gate before the answer reaches the user.
    Call this instead of score_answer() in the full pipeline.
    """
    scored = score_answer(query, verified_result, chunks)
    composite_score = scored["scores"]["composite"]["score"]

    if composite_score < COMPOSITE_THRESHOLD:
        print(f"\n  [Scorer] Composite {composite_score:.2f} below "
              f"threshold {COMPOSITE_THRESHOLD} — returning graceful refusal")

        refusal = build_graceful_refusal(
            query,
            chunks,
            scored["scores"],
        )
        return refusal

    scored["is_refusal"] = False
    return scored