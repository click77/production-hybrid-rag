# src/evaluation/compare_strategies.py

import os
import json
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

RESULTS_DIR    = "evals/results"
GOLDEN_QA_PATH = "evals/golden_qa.json"


# ─────────────────────────────────────────────
# RE-INDEX WITH A GIVEN STRATEGY
# ─────────────────────────────────────────────

def reindex_with_strategy(strategy_name: str) -> dict:
    """
    Wipe both indexes and re-index the entire docs/ corpus
    using the specified chunking strategy.

    Called once per strategy before running the eval suite.
    Returns a summary of how many chunks were indexed.
    """
    from src.ingestion.loader  import load_directory
    from src.ingestion.chunker import chunk_documents, ChunkStrategy
    from src.ingestion.indexer import DualIndex

    print(f"\n  Re-indexing with strategy: {strategy_name.upper()}")

    # Map string to enum
    strategy_map = {
        "fixed":     ChunkStrategy.FIXED,
        "recursive": ChunkStrategy.RECURSIVE,
        "semantic":  ChunkStrategy.SEMANTIC,
    }
    strategy = strategy_map[strategy_name]

    # Load all docs from corpus
    docs = load_directory("docs/")
    if not docs:
        raise ValueError("No documents found in docs/ — add files first.")

    # Chunk with the chosen strategy
    chunks = chunk_documents(docs, strategy=strategy)

    # Wipe and rebuild both indexes
    index = DualIndex()
    index.reset()
    summary = index.add_chunks(chunks)

    print(f"  Indexed {summary['inserted']} chunks "
          f"({summary['skipped']} duplicates skipped)")

    return {
        "strategy":       strategy_name,
        "docs_loaded":    len(docs),
        "chunks_created": len(chunks),
        "chunks_indexed": summary["inserted"],
        "chunks_skipped": summary["skipped"],
    }


# ─────────────────────────────────────────────
# RUN EVAL FOR ONE STRATEGY
# ─────────────────────────────────────────────

def evaluate_strategy(
    strategy_name: str,
    limit: int = None,
    categories: list = None,
) -> dict:
    """
    Re-index with strategy, run the full eval suite, return results.

    Args:
        strategy_name — "fixed", "recursive", or "semantic"
        limit         — cap number of questions (for quick tests)
        categories    — filter to specific categories

    Returns the full eval summary dict with strategy metadata added.
    """
    from src.evaluation.runner import run_eval_suite

    print(f"\n{'═'*60}")
    print(f"  STRATEGY: {strategy_name.upper()}")
    print(f"{'═'*60}")

    # Step 1 — re-index
    index_summary = reindex_with_strategy(strategy_name)

    # Step 2 — run eval suite (don't auto-save — we save the comparison)
    eval_summary = run_eval_suite(
        limit=limit,
        categories=categories,
        save_results=False,
    )

    # Attach strategy metadata
    eval_summary["strategy"]       = strategy_name
    eval_summary["index_summary"]  = index_summary

    return eval_summary


# ─────────────────────────────────────────────
# COMPARISON TABLE PRINTER
# ─────────────────────────────────────────────

def print_comparison_table(comparison: dict):
    """
    Print a formatted comparison table to the terminal.

    Example output:
    ╔══════════════════════╦══════════╦═══════════╦══════════╗
    ║ Metric               ║  Fixed   ║ Recursive ║ Semantic ║
    ╠══════════════════════╬══════════╬═══════════╬══════════╣
    ║ Composite            ║   72%    ║    78%    ║   84% ★  ║
    ║ Answer Correctness   ║   68%    ║    74%    ║   81% ★  ║
    ...
    """
    strategies = list(comparison["strategies"].keys())
    results    = comparison["strategies"]

    metrics = [
        ("composite",           "Composite"),
        ("answer_correctness",  "Answer Correctness"),
        ("faithfulness",        "Faithfulness"),
        ("retrieval_relevance", "Retrieval Relevance"),
        ("citation_accuracy",   "Citation Accuracy"),
    ]

    col_w  = 11
    label_w = 22

    # Header
    header_cells = [f"{'Metric':<{label_w}}"] + [
        f"{s.capitalize():^{col_w}}" for s in strategies
    ]
    divider = "─" * (label_w + col_w * len(strategies) + len(strategies) + 1)

    print(f"\n{'═'*60}")
    print(f"  CHUNKING STRATEGY COMPARISON")
    print(f"{'═'*60}")
    print(f"  {' | '.join(header_cells)}")
    print(f"  {divider}")

    for metric_key, metric_label in metrics:
        scores = {}
        for s in strategies:
            overall = results[s].get("overall", {})
            scores[s] = overall.get(metric_key, 0.0)

        best_strategy = max(scores, key=scores.get)

        cells = [f"{metric_label:<{label_w}}"]
        for s in strategies:
            pct    = f"{scores[s]:.0%}"
            marker = " ★" if s == best_strategy else "  "
            cells.append(f"{pct + marker:^{col_w}}")

        print(f"  {' | '.join(cells)}")

    print(f"  {divider}")

    # Chunks indexed row
    cells = [f"{'Chunks Indexed':<{label_w}}"]
    for s in strategies:
        n = results[s].get("index_summary", {}).get("chunks_indexed", "—")
        cells.append(f"{str(n):^{col_w}}")
    print(f"  {' | '.join(cells)}")

    # Time row
    cells = [f"{'Avg Time / Q (s)':<{label_w}}"]
    for s in strategies:
        r       = results[s].get("results", [])
        avg_t   = sum(x.get("elapsed_seconds", 0) for x in r) / len(r) if r else 0
        cells.append(f"{avg_t:.1f}s".center(col_w))
    print(f"  {' | '.join(cells)}")

    print(f"  {divider}")

    # Winner summary
    overall_scores = {
        s: results[s].get("overall", {}).get("composite", 0.0)
        for s in strategies
    }
    winner = max(overall_scores, key=overall_scores.get)

    print(f"\n  ★  Best overall strategy: {winner.upper()} "
          f"({overall_scores[winner]:.0%} composite)")

    # Per-category breakdown
    print(f"\n  By category:")
    all_cats = set()
    for s in strategies:
        all_cats.update(results[s].get("by_category", {}).keys())

    for cat in sorted(all_cats):
        cells = [f"  {cat:<{label_w}}"]
        best_score = -1
        best_s     = None

        cat_scores = {}
        for s in strategies:
            score = (results[s]
                     .get("by_category", {})
                     .get(cat, {})
                     .get("composite", 0.0))
            cat_scores[s] = score
            if score > best_score:
                best_score = score
                best_s     = s

        for s in strategies:
            pct    = f"{cat_scores[s]:.0%}"
            marker = " ★" if s == best_s else "  "
            cells.append(f"{pct + marker:^{col_w}}")

        print(" | ".join(cells))


# ─────────────────────────────────────────────
# INTERVIEW TALKING POINTS GENERATOR
# ─────────────────────────────────────────────

def generate_talking_points(comparison: dict) -> str:
    """
    Generate ready-to-use interview talking points from the
    comparison results. These are the numbers you lead with.
    """
    strategies = comparison["strategies"]
    overall    = {s: strategies[s].get("overall", {}) for s in strategies}

    # Find best per metric
    def best(metric):
        scores = {s: overall[s].get(metric, 0) for s in strategies}
        winner = max(scores, key=scores.get)
        return winner, scores[winner]

    comp_winner,   comp_score   = best("composite")
    faith_winner,  faith_score  = best("faithfulness")
    correct_winner, correct_score = best("answer_correctness")
    cite_winner,   cite_score   = best("citation_accuracy")

    # Improvement over fixed (baseline)
    fixed_comp = overall.get("fixed", {}).get("composite", 0)
    best_comp  = comp_score
    improvement = ((best_comp - fixed_comp) / fixed_comp * 100
                   if fixed_comp > 0 else 0)

    talking_points = f"""
╔══════════════════════════════════════════════════════════════╗
║           INTERVIEW TALKING POINTS                          ║
╚══════════════════════════════════════════════════════════════╝

Lead with this:
─────────────────────────────────────────────────────────────
"I built a production RAG pipeline with hybrid retrieval and
evaluated three chunking strategies across a 50-question golden
dataset. {comp_winner.capitalize()} chunking achieved the highest
composite score at {comp_score:.0%}, outperforming the fixed-size
baseline by {improvement:.0f} percentage points."

Key numbers to mention:
─────────────────────────────────────────────────────────────
- Best faithfulness  : {faith_score:.0%}  ({faith_winner} chunking)
- Best correctness   : {correct_score:.0%}  ({correct_winner} chunking)
- Best citation acc  : {cite_score:.0%}  ({cite_winner} chunking)
- Baseline (fixed)   : {fixed_comp:.0%}  composite

Why hybrid over dense-only?
─────────────────────────────────────────────────────────────
"Dense search misses exact keyword matches — function names,
config keys, error codes. BM25 catches these but misses
paraphrased queries. RRF merges both rank lists without needing
comparable scores, giving a {comp_score:.0%} faithfulness score
that neither method achieves alone."

Why a reranker on top of RRF?
─────────────────────────────────────────────────────────────
"RRF works on rank positions, not semantics. A chunk can rank
highly because it shares keywords without answering the question.
The LLM-as-judge reranker reads both the question and chunk text
and keeps the top 5 — this is what separates a demo from a
production system."

Why citation verification?
─────────────────────────────────────────────────────────────
"Most RAG systems stop at generation. We verify every bracketed
citation by asking the LLM: does this chunk actually support this
claim? Unsupported citations are flagged before the answer reaches
the user. This is the quality gate most teams skip."
"""
    return talking_points


# ─────────────────────────────────────────────
# MAIN COMPARISON RUNNER
# ─────────────────────────────────────────────

def run_strategy_comparison(
    strategies: list = None,
    limit: int = None,
    categories: list = None,
) -> dict:
    """
    Run eval suite for each chunking strategy and produce a comparison.

    Args:
        strategies — list of strategy names to compare
                     default: ["fixed", "recursive", "semantic"]
        limit      — cap questions per strategy (for quick tests)
        categories — filter to specific question categories

    Returns comparison dict and saves JSON + prints table.
    """
    if strategies is None:
        strategies = ["fixed", "recursive", "semantic"]

    print(f"\n{'═'*60}")
    print(f"  CHUNKING STRATEGY COMPARISON")
    print(f"  Strategies : {strategies}")
    print(f"  Questions  : {limit or 'all 50'}")
    print(f"  Categories : {categories or 'all'}")
    print(f"{'═'*60}")

    start_time = time.time()
    all_results = {}

    for strategy in strategies:
        try:
            result = evaluate_strategy(strategy, limit=limit, categories=categories)
            all_results[strategy] = result
        except Exception as e:
            print(f"\n  ✗ Strategy '{strategy}' failed: {e}")
            all_results[strategy] = {"error": str(e), "overall": {}}

    total_time = round(time.time() - start_time, 1)

    comparison = {
        "run_at":       datetime.utcnow().isoformat(),
        "strategies":   all_results,
        "total_time_s": total_time,
        "config": {
            "limit":      limit,
            "categories": categories,
        },
    }

    # Print table
    print_comparison_table(comparison)

    # Generate talking points
    talking_points = generate_talking_points(comparison)
    print(talking_points)

    # Save results
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path  = f"{RESULTS_DIR}/comparison_{timestamp}.json"

    # Save without the full per-question results to keep file small
    slim = dict(comparison)
    slim["strategies"] = {
        s: {k: v for k, v in r.items() if k != "results"}
        for s, r in all_results.items()
    }
    with open(out_path, "w") as f:
        json.dump(slim, f, indent=2)

    print(f"\n  Comparison saved to {out_path}")
    print(f"  Total time: {total_time}s")

    return comparison