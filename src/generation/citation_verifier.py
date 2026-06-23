# src/generation/citation_verifier.py

import os
import re
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

VERIFIER_MODEL = "gpt-4o-mini"   # fast + cheap for binary judgments


# ─────────────────────────────────────────────
# CITATION PARSER
# ─────────────────────────────────────────────

def parse_citation_pairs(answer: str, chunks: list) -> list:
    """
    Split the answer into sentences and extract which chunk numbers
    each sentence cites.

    Returns a list of dicts:
        {
            "sentence":      "BM25 uses term frequency. [2]",
            "claim":         "BM25 uses term frequency.",   # text without citation
            "citation_nums": [2],                           # which chunks cited
            "chunks":        [<chunk dict for chunk 2>]     # the actual chunks
        }

    Only returns sentences that contain at least one citation.
    Sentences without citations are not verified (they shouldn't exist
    given the system prompt, but we skip them gracefully).
    """
    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())

    pairs = []
    for sentence in sentences:
        # Find all citation numbers in this sentence e.g. [1][3] → [1, 3]
        citation_nums = [int(n) for n in re.findall(r"\[(\d+)\]", sentence)]

        if not citation_nums:
            continue

        # Strip the citation brackets to get just the claim text
        claim = re.sub(r"\[\d+\]", "", sentence).strip()

        # Look up the actual chunk dicts for each citation number
        cited_chunks = []
        for num in citation_nums:
            idx = num - 1    # citations are 1-indexed, list is 0-indexed
            if 0 <= idx < len(chunks):
                cited_chunks.append(chunks[idx])

        if claim and cited_chunks:
            pairs.append({
                "sentence":      sentence,
                "claim":         claim,
                "citation_nums": citation_nums,
                "chunks":        cited_chunks,
            })

    return pairs


# ─────────────────────────────────────────────
# LLM-AS-JUDGE VERIFICATION
# ─────────────────────────────────────────────

def _verify_single_pair(claim: str, chunk_text: str, citation_num: int) -> dict:
    """
    Ask GPT-4o-mini to judge whether a chunk actually supports a claim.

    Returns a dict:
        {
            "supported":    True/False,
            "confidence":   0.0–1.0,
            "reasoning":    "one sentence explanation",
            "citation_num": int
        }
    """
    prompt = f"""You are a citation verification system for a RAG pipeline.

Your job: determine whether the CHUNK TEXT actually supports the CLAIM.

CLAIM:
{claim}

CHUNK TEXT (citation [{citation_num}]):
{chunk_text[:800]}

Rules:
- "supported" = True only if the chunk text directly and explicitly
  supports the claim. Indirect or tangential support = False.
- "confidence" = your confidence in the judgment (0.0 to 1.0)
- "reasoning"  = one sentence explaining your decision

Respond with ONLY valid JSON in this exact format:
{{
  "supported":  true,
  "confidence": 0.95,
  "reasoning":  "The chunk explicitly states that BM25 uses term frequency."
}}"""

    response = client.chat.completions.create(
        model=VERIFIER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=150,
    )

    raw = response.choices[0].message.content.strip()

    try:
        result = json.loads(raw)
        return {
            "supported":    bool(result.get("supported", False)),
            "confidence":   float(result.get("confidence", 0.0)),
            "reasoning":    result.get("reasoning", ""),
            "citation_num": citation_num,
        }
    except (json.JSONDecodeError, ValueError):
        # If parsing fails, default to unsupported
        print(f"  [Verifier] Warning: could not parse response: {raw}")
        return {
            "supported":    False,
            "confidence":   0.0,
            "reasoning":    f"Could not parse verifier response: {raw}",
            "citation_num": citation_num,
        }


# ─────────────────────────────────────────────
# FULL VERIFICATION PIPELINE
# ─────────────────────────────────────────────

def verify_citations(generation_result: dict) -> dict:
    """
    Verify every citation in a generated answer.

    Takes the full output dict from generate_answer() and checks
    whether each bracketed citation actually supports the claim
    it is attached to.

    For each citation-claim pair:
        1. Extract the claim (sentence without brackets)
        2. Look up the cited chunk text
        3. Ask GPT-4o-mini: does this chunk support this claim?
        4. Flag unsupported citations

    Returns the generation_result dict with these fields added:
        verification_results  — list of per-citation verdicts
        verified_citations    — citation numbers that passed
        unsupported_citations — citation numbers that failed
        verification_score    — % of citations that are supported
        has_hallucinations    — True if any citation failed
        flagged_answer        — answer text with [UNSUPPORTED] markers
    """
    answer = generation_result.get("answer", "")
    chunks = generation_result.get("chunks", [])

    print(f"\n  [Verifier] Parsing citations from answer...")

    pairs = parse_citation_pairs(answer, chunks)

    if not pairs:
        print("  [Verifier] No citation pairs found to verify.")
        return {
            **generation_result,
            "verification_results":  [],
            "verified_citations":    [],
            "unsupported_citations": [],
            "verification_score":    1.0,
            "has_hallucinations":    False,
            "flagged_answer":        answer,
        }

    print(f"  [Verifier] Found {len(pairs)} citation pairs to verify...")

    verification_results  = []
    verified_citations    = set()
    unsupported_citations = set()

    for pair in pairs:
        claim         = pair["claim"]
        citation_nums = pair["citation_nums"]
        cited_chunks  = pair["chunks"]

        for chunk, citation_num in zip(cited_chunks, citation_nums):
            print(f"  [Verifier] Checking [{citation_num}] for: "
                  f"'{claim[:60]}...'")

            result = _verify_single_pair(
                claim=claim,
                chunk_text=chunk["text"],
                citation_num=citation_num,
            )

            result["claim"]    = claim
            result["sentence"] = pair["sentence"]
            verification_results.append(result)

            if result["supported"]:
                verified_citations.add(citation_num)
                print(f"    ✓ [{citation_num}] SUPPORTED "
                      f"(confidence: {result['confidence']})")
            else:
                unsupported_citations.add(citation_num)
                print(f"    ✗ [{citation_num}] UNSUPPORTED — "
                      f"{result['reasoning']}")

    # Calculate verification score
    total_checked     = len(verification_results)
    total_supported   = len([r for r in verification_results if r["supported"]])
    verification_score = total_supported / total_checked if total_checked > 0 else 1.0

    # Build flagged answer — mark unsupported citations inline
    flagged_answer = answer
    for num in sorted(unsupported_citations, reverse=True):
        flagged_answer = flagged_answer.replace(
            f"[{num}]",
            f"[{num}][UNSUPPORTED]"
        )

    has_hallucinations = len(unsupported_citations) > 0

    print(f"\n  [Verifier] Score: {verification_score:.0%} "
          f"({total_supported}/{total_checked} citations supported)")

    if has_hallucinations:
        print(f"  [Verifier] ⚠ Unsupported citations: "
              f"{sorted(unsupported_citations)}")
    else:
        print(f"  [Verifier] ✓ All citations verified")

    return {
        **generation_result,
        "verification_results":  verification_results,
        "verified_citations":    sorted(verified_citations),
        "unsupported_citations": sorted(unsupported_citations),
        "verification_score":    round(verification_score, 4),
        "has_hallucinations":    has_hallucinations,
        "flagged_answer":        flagged_answer,
    }