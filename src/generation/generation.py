# src/generation/generator.py

import os
import re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

GENERATION_MODEL = "gpt-4o"


# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise question-answering assistant backed by a \
retrieval system.

You will be given a user question and a set of numbered context chunks \
retrieved from a document corpus.

STRICT RULES you must follow:

1. ANSWER ONLY FROM CONTEXT
   - Base every claim exclusively on the provided context chunks.
   - Never use outside knowledge, assumptions, or training data.

2. CITE EVERY CLAIM
   - After every sentence that makes a factual claim, add a bracketed
     citation referencing the chunk number(s) that support it.
   - Format: [1], [2], [1][3]
   - Example: "BM25 uses term frequency to score documents. [2]"

3. WHEN CONTEXT IS INSUFFICIENT
   - If the context does not contain enough information to answer, say
     exactly this and nothing more:
     "The provided context does not contain enough information to answer
      this question. You may want to check: <suggest relevant topic>"
   - Never fabricate an answer when context is missing.

4. STAY GROUNDED
   - Do not infer, extrapolate, or combine context with outside knowledge.
   - If only part of the question can be answered from context, answer
     that part with citations and flag what could not be answered.

5. FORMAT
   - Write in clear, concise prose.
   - Use bullet points only if the question explicitly asks for a list.
   - Keep answers focused — do not pad with unnecessary context."""


# ─────────────────────────────────────────────
# CONTEXT BLOCK BUILDER
# ─────────────────────────────────────────────

def build_context_blocks(chunks: list) -> str:
    """
    Format retrieved chunks into numbered context blocks
    that are passed in the user turn of the prompt.

    Each block includes:
    - chunk number (used for citations)
    - source file and page number
    - section heading if available
    - the chunk text

    Example output:
        [1] Source: docs/intro.md | Page: 1 | Section: Introduction
        BM25 is a ranking function used in information retrieval...

        [2] Source: docs/retrieval.pdf | Page: 3 | Section: Dense Search
        Dense retrieval embeds both query and documents...
    """
    blocks = []

    for i, chunk in enumerate(chunks, start=1):
        source   = chunk.get("source_file", "unknown")
        page     = chunk.get("page_number", 1)
        heading  = chunk.get("section_heading") or "—"
        text     = chunk.get("text", "").strip()

        block = (
            f"[{i}] Source: {source} | Page: {page} | Section: {heading}\n"
            f"{text}"
        )
        blocks.append(block)

    return "\n\n".join(blocks)


def build_user_prompt(query: str, chunks: list) -> str:
    """
    Build the full user turn: context blocks + the question.
    Keeping context in the user turn (not system) gives the model
    clearer separation between instructions and retrieved content.
    """
    context = build_context_blocks(chunks)

    return f"""CONTEXT:
{context}

─────────────────────────────
QUESTION:
{query}

Answer using only the context above. Cite every claim with [chunk number]."""


# ─────────────────────────────────────────────
# GENERATION
# ─────────────────────────────────────────────

def generate_answer(
    query: str,
    chunks: list,
    model: str = GENERATION_MODEL,
    temperature: float = 0.1,
    max_tokens: int = 1024,
) -> dict:
    """
    Generate a grounded, cited answer from retrieved chunks.

    Args:
        query       — the user's original question
        chunks      — reranked chunk dicts from retrieve_and_rerank()
        model       — OpenAI model to use (default gpt-4o)
        temperature — low temperature for factual consistency
        max_tokens  — max length of generated answer

    Returns a dict containing:
        answer          — the generated answer text with citations
        query           — original question
        model           — model used
        chunks_used     — number of context chunks passed in
        context_blocks  — the formatted context string (for debugging)
        citations_found — list of citation numbers found in the answer
        finish_reason   — why generation stopped (stop / length)
    """
    if not chunks:
        return {
            "answer":          "No context chunks were retrieved for this query.",
            "query":           query,
            "model":           model,
            "chunks_used":     0,
            "context_blocks":  "",
            "citations_found": [],
            "finish_reason":   "no_context",
        }

    print(f"\n  [Generator] Generating answer with {len(chunks)} chunks...")
    print(f"  [Generator] Model: {model}")

    user_prompt    = build_user_prompt(query, chunks)
    context_blocks = build_context_blocks(chunks)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    answer        = response.choices[0].message.content.strip()
    finish_reason = response.choices[0].finish_reason

    # Extract all citation numbers found in the answer e.g. [1], [2][3]
    citations_found = list(set(
        int(n) for n in re.findall(r"\[(\d+)\]", answer)
    ))
    citations_found.sort()

    print(f"  [Generator] Answer generated ({len(answer)} chars)")
    print(f"  [Generator] Citations found: {citations_found}")
    print(f"  [Generator] Finish reason: {finish_reason}")

    return {
        "answer":          answer,
        "query":           query,
        "model":           model,
        "chunks_used":     len(chunks),
        "context_blocks":  context_blocks,
        "citations_found": citations_found,
        "finish_reason":   finish_reason,
        "chunks":          chunks,   # pass through for citation verifier
    }


# ─────────────────────────────────────────────
# INSUFFICIENT CONTEXT DETECTION
# ─────────────────────────────────────────────

def is_insufficient_context(answer: str) -> bool:
    """
    Detect whether the generator flagged that context was insufficient.
    Used downstream to decide whether to surface a fallback response.
    """
    signal = "does not contain enough information"
    return signal.lower() in answer.lower()