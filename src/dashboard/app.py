# src/dashboard/app.py

import os
import sys
import json
import requests
from pathlib import Path

import streamlit as st

# Make sure src/ is importable when running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

API_URL = os.getenv("API_URL", "http://localhost:8000")


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Hybrid RAG Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .chunk-box {
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
        font-size: 0.875rem;
    }
    .score-pill {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .score-high   { background: #1a472a; color: #6fcf97; }
    .score-medium { background: #3d2b00; color: #f2a03f; }
    .score-low    { background: #4a1a1a; color: #eb5757; }
    .citation-tag {
        background: #2d3561;
        color: #7aa2f7;
        border-radius: 4px;
        padding: 1px 6px;
        font-size: 0.78rem;
        font-weight: 700;
        margin: 0 2px;
    }
    .unsupported-tag {
        background: #4a1a1a;
        color: #eb5757;
        border-radius: 4px;
        padding: 1px 6px;
        font-size: 0.78rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def score_color(score: float) -> str:
    if score >= 0.75: return "score-high"
    if score >= 0.50: return "score-medium"
    return "score-low"


def score_emoji(score: float) -> str:
    if score >= 0.75: return "🟢"
    if score >= 0.50: return "🟡"
    return "🔴"


def call_api(endpoint: str, method: str = "GET", payload: dict = None) -> dict:
    try:
        url = f"{API_URL}{endpoint}"
        if method == "POST":
            r = requests.post(url, json=payload, timeout=120)
        else:
            r = requests.get(url, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Make sure the FastAPI server is running: "
                 "`uvicorn src.api.main:app --reload`")
        return {}
    except Exception as e:
        st.error(f"API error: {e}")
        return {}


def render_answer(answer: str, unsupported: list):
    """Render the answer with colour-coded citation tags."""
    if not answer:
        return

    import re
    parts = re.split(r"(\[\d+\](?:\[UNSUPPORTED\])?)", answer)
    rendered = ""
    for part in parts:
        num_match = re.match(r"\[(\d+)\](\[UNSUPPORTED\])?", part)
        if num_match:
            num      = int(num_match.group(1))
            is_bad   = num_match.group(2) or num in unsupported
            css      = "unsupported-tag" if is_bad else "citation-tag"
            label    = f"[{num}]{'⚠' if is_bad else ''}"
            rendered += f'<span class="{css}">{label}</span>'
        else:
            rendered += part.replace("\n", "<br>")

    st.markdown(rendered, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")

    st.subheader("Retrieval")
    dense_weight  = st.slider("Dense weight",  0.0, 1.0, 0.7, 0.05)
    sparse_weight = round(1.0 - dense_weight, 2)
    st.caption(f"Sparse weight: {sparse_weight}")

    top_k        = st.slider("Final chunks (reranker top-k)", 1, 10, 5)
    fusion_top_k = st.slider("Fusion candidates",            10, 40, 20)

    st.subheader("Generation")
    model = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini"])
    verify_citations = st.toggle("Verify citations", value=True)

    st.subheader("Display")
    compare_mode = st.toggle(
        "Compare hybrid vs dense-only",
        value=False,
        help="Run the same query twice — once with hybrid retrieval, "
             "once with dense-only — and show results side by side.",
    )

    st.divider()
    st.subheader("Index Status")
    if st.button("Refresh status"):
        health = call_api("/health")
        if health:
            st.metric("Chunks indexed", health.get("chunk_count", 0))
            status = health.get("index_status", "unknown")
            st.success(status) if status == "ready" else st.error(status)

    st.divider()
    st.subheader("Ingest Document")
    uploaded = st.file_uploader(
        "Upload .txt / .md / .html / .pdf",
        type=["txt", "md", "html", "pdf"],
    )
    ingest_strategy = st.selectbox(
        "Chunking strategy",
        ["recursive", "fixed", "semantic"],
    )

    if uploaded and st.button("Index document"):
        with st.spinner("Ingesting..."):
            files = {"file": (uploaded.name, uploaded.getvalue())}
            try:
                r = requests.post(
                    f"{API_URL}/v1/ingest?strategy={ingest_strategy}",
                    files=files,
                    timeout=120,
                )
                result = r.json()
                st.success(
                    f"Indexed {result.get('chunks_indexed')} chunks "
                    f"({result.get('duplicates_skipped')} duplicates skipped)"
                )
            except Exception as e:
                st.error(f"Ingest failed: {e}")


# ─────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────

st.title("🔍 Hybrid RAG Dashboard")
st.caption("Dense + Sparse retrieval · RRF Fusion · LLM Reranking · "
           "Citation Verification · Confidence Scoring")

query = st.text_input(
    "Ask a question",
    placeholder="How does hybrid retrieval work?",
)

ask_btn = st.button("Ask", type="primary", use_container_width=True)

if ask_btn and query.strip():

    # ── Build request payload ─────────────────────────────────────────
    payload = {
        "question":         query,
        "top_k":            top_k,
        "fusion_top_k":     fusion_top_k,
        "dense_weight":     dense_weight,
        "sparse_weight":    sparse_weight,
        "verify_citations": verify_citations,
        "model":            model,
    }

    if compare_mode:
        # Run hybrid and dense-only in parallel columns
        col_hybrid, col_dense = st.columns(2)

        with col_hybrid:
            st.subheader("🔀 Hybrid (Dense + Sparse)")
            with st.spinner("Running hybrid retrieval..."):
                hybrid_result = call_api("/v1/ask", "POST", payload)

        dense_payload = {**payload, "dense_weight": 1.0, "sparse_weight": 0.0}
        with col_dense:
            st.subheader("📐 Dense Only")
            with st.spinner("Running dense-only retrieval..."):
                dense_result = call_api("/v1/ask", "POST", dense_payload)

        # Render both side by side
        for col, result, label in [
            (col_hybrid, hybrid_result, "Hybrid"),
            (col_dense,  dense_result,  "Dense"),
        ]:
            with col:
                if not result:
                    continue

                scores      = result.get("scores", {})
                composite   = scores.get("composite", {})
                comp_score  = composite.get("score", 0)
                unsupported = result.get("unsupported_citations", [])

                # Composite score
                st.metric(
                    "Composite score",
                    f"{comp_score:.0%}",
                    delta=f"{composite.get('quality', '')}",
                )

                if result.get("is_refusal"):
                    st.warning("⚠ Graceful refusal triggered")

                st.markdown("**Answer**")
                render_answer(result.get("answer", ""), unsupported)

                # Score breakdown
                with st.expander("Score breakdown"):
                    for dim in ["retrieval_confidence",
                                "citation_coverage",
                                "answer_completeness"]:
                        dim_data = scores.get(dim, {})
                        s        = dim_data.get("score", 0)
                        st.markdown(
                            f"{score_emoji(s)} **{dim.replace('_', ' ').title()}**: "
                            f"`{s:.0%}` — {dim_data.get('reasoning', '')}"
                        )

    else:
        # Single query mode
        with st.spinner("Retrieving and generating..."):
            result = call_api("/v1/ask", "POST", payload)

        if not result:
            st.stop()

        scores      = result.get("scores", {})
        composite   = scores.get("composite", {})
        comp_score  = composite.get("score", 0)
        unsupported = result.get("unsupported_citations", [])
        chunks      = result.get("chunks", [])

        # ── Top metrics row ───────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Composite",   f"{comp_score:.0%}",
                  composite.get("quality", ""))
        m2.metric("Retrieval",
                  f"{scores.get('retrieval_confidence', {}).get('score', 0):.0%}")
        m3.metric("Citation",
                  f"{scores.get('citation_coverage', {}).get('score', 0):.0%}")
        m4.metric("Completeness",
                  f"{scores.get('answer_completeness', {}).get('score', 0):.0%}")

        st.divider()

        # ── Main content: answer + chunks ─────────────────────────────
        ans_col, chunk_col = st.columns([3, 2])

        with ans_col:
            st.subheader("Answer")

            if result.get("is_refusal"):
                st.warning("⚠ Graceful refusal — insufficient context")

            if result.get("has_hallucinations"):
                st.error(
                    f"⚠ {len(unsupported)} unsupported citation(s) detected: "
                    f"{unsupported}"
                )

            render_answer(result.get("answer", ""), unsupported)

            # Citation verification detail
            if verify_citations:
                with st.expander("Citation verification detail"):
                    verified   = result.get("verified_citations", [])
                    unverified = result.get("unsupported_citations", [])

                    if verified:
                        st.success(f"✓ Verified citations: {verified}")
                    if unverified:
                        st.error(f"✗ Unsupported citations: {unverified}")

                    halluc = result.get("has_hallucinations", False)
                    st.markdown(
                        f"**Hallucination detected:** {'Yes ⚠' if halluc else 'No ✓'}"
                    )

            # Score breakdown
            with st.expander("Confidence score breakdown"):
                for dim in ["retrieval_confidence",
                            "citation_coverage",
                            "answer_completeness"]:
                    dim_data = scores.get(dim, {})
                    s        = dim_data.get("score", 0)
                    st.markdown(
                        f"{score_emoji(s)} **{dim.replace('_',' ').title()}**: "
                        f"`{s:.0%}`"
                    )
                    st.caption(dim_data.get("reasoning", ""))

                    # Show signals if available
                    signals = dim_data.get("signals", {})
                    if signals:
                        st.json(signals, expanded=False)

        with chunk_col:
            st.subheader(f"Retrieved Chunks ({len(chunks)})")
            st.caption("Ranked by reranker score · click to expand")

            for i, chunk in enumerate(chunks):
                rerank = chunk.get("rerank_score", 0) or 0
                rrf    = chunk.get("rrf_score",    0) or 0
                source = Path(chunk.get("source_file", "unknown")).name
                page   = chunk.get("page_number", 1)
                heading= chunk.get("section_heading") or "—"

                with st.expander(
                    f"[{i+1}] {source} · p{page} · "
                    f"rerank={rerank:.1f}/10"
                ):
                    score_col1, score_col2 = st.columns(2)
                    score_col1.metric("Rerank score", f"{rerank:.1f}/10")
                    score_col2.metric("RRF score",    f"{rrf:.4f}")

                    st.caption(f"Section: {heading}")
                    st.caption(f"Strategy: {chunk.get('strategy', '—')} · "
                               f"Tokens: {chunk.get('token_count', '—')}")
                    st.divider()
                    st.markdown(chunk.get("text", ""))

        # ── Raw JSON tab ──────────────────────────────────────────────
        with st.expander("Raw API response (JSON)"):
            st.json(result)


# ─────────────────────────────────────────────
# DOCUMENTS TAB
# ─────────────────────────────────────────────

st.divider()
with st.expander("📄 View indexed documents"):
    if st.button("Load document list"):
        docs_result = call_api("/v1/documents")
        if docs_result:
            st.metric("Total chunks", docs_result.get("total_chunks", 0))
            sources = docs_result.get("unique_sources", [])
            st.markdown(f"**Unique sources ({len(sources)}):**")
            for s in sources:
                st.markdown(f"- `{s}`")

            docs = docs_result.get("documents", [])
            if docs:
                import pandas as pd
                df = pd.DataFrame(docs)
                st.dataframe(df, use_container_width=True)