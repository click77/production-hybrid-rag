# src/evaluation/runner.py

import os
import json
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EVAL_MODEL     = "gpt-4o-mini"
GOLDEN_QA_PATH = "evals/golden_qa.json"
RESULTS_DIR    = "evals/results"


# ─────────────────────────────────────────────
# LOAD GOLDEN DATASET
# ─────────────────────────────────────────────

def load_golden_qa(path: str = GOLDEN_QA_PATH) -> list:
    """Load the hand-written golden Q&A dataset."""
    with open(path, "r") as f:
        data = json.load(f)
    questions = data["questions"]
    print(f"  Loaded {len(questions)} golden Q&A pairs")
    return questions


# ─────────────────────────────────────────────
# METRIC 1 — ANSWER CORRECTNESS
# ─────────────────────────────────────────────

def metric_answer_correctness(
    question: str,
    generated_answer: str,
    golden_answer: str,
) -> dict:
    """
    LLM-as-judge: does the generated answer match the golden answer?

    Scores 0.0–1.0:
        1.0 = fully correct, same meaning as golden
        0.7 = mostly correct, minor omissions
        0.4 = partially correct, missing key points
        0.1 = incorrect or hallucinated
        0.0 = completely wrong or refused when answer exists

    Special case: if the golden answer starts with UNANSWERABLE,
    we check that the system correctly refused rather than hallucinating.
    """
    is_unanswerable = golden_answer.strip().startswith("UNANSWERABLE")

    if is_unanswerable:
        prompt = f"""You are evaluating a RAG system's ability to refuse unanswerable questions.

QUESTION: {question}
SYSTEM RESPONSE: {generated_answer}

The correct behavior is to refuse to answer because the information
is not in the document corpus.

Did the system correctly refuse rather than hallucinate an answer?
Respond with ONLY valid JSON:
{{
  "score": 0.9,
  "correct_refusal": true,
  "reasoning": "System clearly stated it could not answer and did not fabricate."
}}"""
    else:
        prompt = f"""You are evaluating whether a RAG system's answer is correct.

QUESTION: {question}
GOLDEN ANSWER: {golden_answer}
GENERATED ANSWER: {generated_answer}

Score the generated answer from 0.0 to 1.0 based on factual correctness
compared to the golden answer. Partial credit is allowed.

Respond with ONLY valid JSON:
{{
  "score": 0.85,
  "reasoning": "Generated answer covers the main points but omits X."
}}"""

    response = client.chat.completions.create(
        model=EVAL_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=200,
    )

    try:
        result = json.loads(response.choices[0].message.content.strip())
        return {
            "score":     round(float(result.get("score", 0.0)), 4),
            "reasoning": result.get("reasoning", ""),
            "metric":    "answer_correctness",
        }
    except Exception:
        return {"score": 0.0, "reasoning": "Parse error", "metric": "answer_correctness"}


# ─────────────────────────────────────────────
# METRIC 2 — FAITHFULNESS
# ─────────────────────────────────────────────

def metric_faithfulness(
    generated_answer: str,
    chunks: list,
) -> dict:
    """
    Are all claims in the generated answer grounded in the retrieved context?

    Faithfulness checks that the generator didn't introduce facts
    from outside the provided chunks — i.e. no hallucination.

    Score 0.0–1.0:
        1.0 = every claim is directly supported by context
        0.5 = some claims are unsupported or extrapolated
        0.0 = answer is mostly fabricated
    """
    if not chunks:
        return {
            "score": 0.0,
            "reasoning": "No chunks retrieved — cannot assess faithfulness.",
            "metric": "faithfulness",
        }

    context = "\n\n".join(
        f"[{i+1}] {c['text'][:500]}" for i, c in enumerate(chunks)
    )

    prompt = f"""You are evaluating the faithfulness of a RAG system's answer.

Faithfulness means: every factual claim in the answer is directly
supported by the provided context. No outside knowledge should be used.

CONTEXT CHUNKS:
{context}

GENERATED ANSWER:
{generated_answer}

Score faithfulness from 0.0 to 1.0.
List any claims that are NOT supported by the context.

Respond with ONLY valid JSON:
{{
  "score": 0.9,
  "unsupported_claims": ["Claim X is not in the context"],
  "reasoning": "Most claims are supported. Claim X was not found."
}}"""

    response = client.chat.completions.create(
        model=EVAL_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=300,
    )

    try:
        result = json.loads(response.choices[0].message.content.strip())
        return {
            "score":             round(float(result.get("score", 0.0)), 4),
            "unsupported_claims": result.get("unsupported_claims", []),
            "reasoning":         result.get("reasoning", ""),
            "metric":            "faithfulness",
        }
    except Exception:
        return {"score": 0.0, "reasoning": "Parse error", "metric": "faithfulness"}


# ─────────────────────────────────────────────
# METRIC 3 — RETRIEVAL RELEVANCE
# ─────────────────────────────────────────────

def metric_retrieval_relevance(
    question: str,
    chunks: list,
    golden_source_section: str = None,
) -> dict:
    """
    Were the right chunks retrieved for this question?

    Two signals:
    1. LLM judge: are the retrieved chunks relevant to the question?
    2. Source match: did retrieval find chunks from the expected section?
       (only checked when golden_source_section is provided)

    Score 0.0–1.0.
    """
    if not chunks:
        return {
            "score":    0.0,
            "reasoning": "No chunks retrieved.",
            "metric":    "retrieval_relevance",
        }

    chunk_previews = "\n".join(
        f"[{i+1}] {c['text'][:300]}..." for i, c in enumerate(chunks[:5])
    )

    prompt = f"""You are evaluating whether retrieved chunks are relevant
to a user's question in a RAG system.

QUESTION: {question}

RETRIEVED CHUNKS (top 5):
{chunk_previews}

Score relevance from 0.0 to 1.0:
  1.0 = chunks directly answer the question
  0.5 = chunks are topically related but not directly useful
  0.0 = chunks are unrelated to the question

Respond with ONLY valid JSON:
{{
  "score": 0.8,
  "reasoning": "Chunks 1 and 2 directly address the question."
}}"""

    response = client.chat.completions.create(
        model=EVAL_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=200,
    )

    try:
        result = json.loads(response.choices[0].message.content.strip())
        llm_score = float(result.get("score", 0.0))

        # Bonus: check if expected source section appears in retrieved chunks
        source_bonus = 0.0
        if golden_source_section:
            for chunk in chunks:
                heading = chunk.get("section_heading", "") or ""
                if golden_source_section.lower() in heading.lower():
                    source_bonus = 0.1
                    break

        final_score = round(min(llm_score + source_bonus, 1.0), 4)

        return {
            "score":               final_score,
            "llm_score":          round(llm_score, 4),
            "source_match_bonus": source_bonus,
            "reasoning":          result.get("reasoning", ""),
            "metric":             "retrieval_relevance",
        }
    except Exception:
        return {"score": 0.0, "reasoning": "Parse error", "metric": "retrieval_relevance"}


# ─────────────────────────────────────────────
# METRIC 4 — CITATION ACCURACY
# ─────────────────────────────────────────────

def metric_citation_accuracy(verified_result: dict) -> dict:
    """
    Do the citations in the answer actually support the claims?

    Uses the output of citation_verifier.verify_citations() directly —
    no additional API calls needed for this metric.

    Score = proportion of verified citations out of all checked.
    """
    verification_results  = verified_result.get("verification_results", [])
    unsupported_citations = verified_result.get("unsupported_citations", [])
    verification_score    = verified_result.get("verification_score", 1.0)

    if not verification_results:
        # No citations to check — could mean no citations were generated
        has_citations = bool(verified_result.get("citations_found"))
        score = 1.0 if not has_citations else 0.5
        return {
            "score":    score,
            "reasoning": "No citation pairs were verified.",
            "metric":   "citation_accuracy",
        }

    return {
        "score":               round(verification_score, 4),
        "unsupported_count":   len(unsupported_citations),
        "unsupported_list":    unsupported_citations,
        "total_checked":       len(verification_results),
        "reasoning": (
            f"{len(verification_results) - len(unsupported_citations)}/"
            f"{len(verification_results)} citations verified"
        ),
        "metric": "citation_accuracy",
    }


# ─────────────────────────────────────────────
# SINGLE TEST CASE RUNNER
# ─────────────────────────────────────────────

def run_single(qa_item: dict) -> dict:
    """
    Run the full eval pipeline for one golden Q&A item.
    Measures all 4 metrics and returns a result dict.
    """
    from src.retrieval.reranker        import retrieve_and_rerank
    from src.generation.generator      import generate_answer
    from src.generation.citation_verifier import verify_citations

    question       = qa_item["question"]
    golden_answer  = qa_item["answer"]
    source_section = qa_item.get("source_section")
    category       = qa_item.get("category", "unknown")
    qa_id          = qa_item.get("id", "unknown")

    print(f"\n  [{qa_id}] {category.upper()}: {question[:70]}...")

    start = time.time()

    # ── Retrieve ──────────────────────────────────────────────────────
    chunks = retrieve_and_rerank(question, fusion_top_k=20, rerank_top_k=5)

    # ── Generate ──────────────────────────────────────────────────────
    generation = generate_answer(question, chunks)

    # ── Verify citations ──────────────────────────────────────────────
    verified = verify_citations(generation)

    generated_answer = verified.get("answer", "")

    # ── Score all 4 metrics ───────────────────────────────────────────
    correctness = metric_answer_correctness(
        question, generated_answer, golden_answer
    )
    faithfulness = metric_faithfulness(generated_answer, chunks)
    relevance    = metric_retrieval_relevance(
        question, chunks, source_section
    )
    citation_acc = metric_citation_accuracy(verified)

    # ── Composite ─────────────────────────────────────────────────────
    composite = round(
        0.30 * correctness["score"]  +
        0.30 * faithfulness["score"] +
        0.20 * relevance["score"]    +
        0.20 * citation_acc["score"],
        4,
    )

    elapsed = round(time.time() - start, 2)

    result = {
        "id":               qa_id,
        "category":         category,
        "question":         question,
        "golden_answer":    golden_answer,
        "generated_answer": generated_answer,
        "metrics": {
            "answer_correctness":  correctness,
            "faithfulness":        faithfulness,
            "retrieval_relevance": relevance,
            "citation_accuracy":   citation_acc,
            "composite":           composite,
        },
        "elapsed_seconds": elapsed,
        "chunks_retrieved": len(chunks),
    }

    print(f"    correctness={correctness['score']:.2f} "
          f"faithfulness={faithfulness['score']:.2f} "
          f"relevance={relevance['score']:.2f} "
          f"citation={citation_acc['score']:.2f} "
          f"composite={composite:.2f} "
          f"({elapsed}s)")

    return result


# ─────────────────────────────────────────────
# FULL EVAL SUITE RUNNER
# ─────────────────────────────────────────────

def run_eval_suite(
    limit: int = None,
    categories: list = None,
    save_results: bool = True,
) -> dict:
    """
    Run the full evaluation suite over the golden Q&A dataset.

    Args:
        limit      — only run the first N questions (useful for quick tests)
        categories — filter to specific categories e.g. ["straightforward"]
        save_results — write results JSON to evals/results/

    Returns a summary dict with per-category and overall scores.
    """
    questions = load_golden_qa()

    # Apply filters
    if categories:
        questions = [q for q in questions if q["category"] in categories]
    if limit:
        questions = questions[:limit]

    print(f"\n{'═'*60}")
    print(f"  EVAL SUITE — {len(questions)} questions")
    print(f"{'═'*60}")

    results      = []
    failed       = []

    for qa_item in questions:
        try:
            result = run_single(qa_item)
            results.append(result)
        except Exception as e:
            print(f"  ✗ Failed [{qa_item.get('id')}]: {e}")
            failed.append({"id": qa_item.get("id"), "error": str(e)})

    # ── Aggregate scores ──────────────────────────────────────────────
    def avg(lst): return round(sum(lst) / len(lst), 4) if lst else 0.0

    all_composites    = [r["metrics"]["composite"] for r in results]
    all_correctness   = [r["metrics"]["answer_correctness"]["score"]  for r in results]
    all_faithfulness  = [r["metrics"]["faithfulness"]["score"]        for r in results]
    all_relevance     = [r["metrics"]["retrieval_relevance"]["score"] for r in results]
    all_citation      = [r["metrics"]["citation_accuracy"]["score"]   for r in results]

    # Per-category breakdown
    category_scores = {}
    for cat in ["straightforward", "multi_hop", "unanswerable", "ambiguous"]:
        cat_results = [r for r in results if r["category"] == cat]
        if cat_results:
            category_scores[cat] = {
                "count":     len(cat_results),
                "composite": avg([r["metrics"]["composite"] for r in cat_results]),
                "correctness": avg([r["metrics"]["answer_correctness"]["score"]
                                    for r in cat_results]),
            }

    summary = {
        "run_at":         datetime.utcnow().isoformat(),
        "total_questions": len(questions),
        "completed":       len(results),
        "failed":          len(failed),
        "overall": {
            "composite":           avg(all_composites),
            "answer_correctness":  avg(all_correctness),
            "faithfulness":        avg(all_faithfulness),
            "retrieval_relevance": avg(all_relevance),
            "citation_accuracy":   avg(all_citation),
        },
        "by_category":  category_scores,
        "failed_items": failed,
        "results":      results,
    }

    # ── Print summary ─────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  EVAL RESULTS")
    print(f"{'═'*60}")
    print(f"  Composite score      : {summary['overall']['composite']:.0%}")
    print(f"  Answer correctness   : {summary['overall']['answer_correctness']:.0%}")
    print(f"  Faithfulness         : {summary['overall']['faithfulness']:.0%}")
    print(f"  Retrieval relevance  : {summary['overall']['retrieval_relevance']:.0%}")
    print(f"  Citation accuracy    : {summary['overall']['citation_accuracy']:.0%}")
    print(f"\n  By category:")
    for cat, scores in category_scores.items():
        print(f"    {cat:<20} composite={scores['composite']:.0%} "
              f"n={scores['count']}")

    # ── Save results ──────────────────────────────────────────────────
    if save_results:
        Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
        timestamp   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = f"{RESULTS_DIR}/eval_{timestamp}.json"
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Results saved to {output_path}")

    return summary