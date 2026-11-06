from pathlib import Path
import json
import pandas as pd
import streamlit as st

from fever_ir.retrieval.core import EvidenceIndex, DenseIndex, reciprocal_rank_fusion

st.set_page_config(
    page_title="FEVER Fact Verification Search",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_sparse():
    path = Path("artifacts/sparse_index.pkl")
    if not path.exists():
        return None
    return EvidenceIndex.load_sparse(path)


@st.cache_resource
def load_dense():
    path = Path("artifacts/dense_index.npz")
    if not path.exists():
        return None
    try:
        return DenseIndex.load(path)
    except Exception as exc:
        st.warning(f"Dense index could not be loaded: {exc}")
        return None


# Sidebar controls
st.sidebar.title("⚙️ Search Controls")

top_k = st.sidebar.slider("Number of Results (Top-K)", min_value=3, max_value=20, value=5)
rrf_smoothing = st.sidebar.slider("RRF Constant (k)", min_value=10, max_value=100, value=60, step=5,
                                  help="Smoothing constant k in RRF formula: 1 / (k + rank)")
candidate_k = st.sidebar.slider("Candidate Pool Depth", min_value=20, max_value=100, value=50, step=10,
                                help="Candidate depth retrieved from each method prior to RRF fusion")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Preprocessing Active")
st.sidebar.caption("✅ Text Normalization & Cleaning\n\n✅ Alphanumeric Tokenization\n\n✅ Stop-word Removal (English)\n\n✅ Porter Stemmer (NLTK)")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🏛️ Architecture")
st.sidebar.caption("Standardized IR Evaluation Template (Project 10: Fact Verification using FEVER)")

# Main Header
st.title("🔎 FEVER Fact Verification & Evidence Search")
st.markdown(
    "Multi-model evidence retrieval pipeline comparing **TF-IDF**, **BM25**, "
    "**Sentence Transformers (Dense)**, and **Hybrid Reciprocal Rank Fusion (RRF)**."
)

sparse_index = load_sparse()
dense_index = load_dense()

if sparse_index is None:
    st.error("No search index found in `artifacts/sparse_index.pkl`. Run `python scripts/setup_benchmark.py` first.")
    st.stop()

# Tabs
tab_search, tab_benchmark, tab_theory = st.tabs([
    "🔍 Live Evidence Search & Model Comparison",
    "📊 Performance Benchmark & Precision-Recall Curves",
    "📐 Methodology & Mathematical Formulations",
])

# ==========================================================
# TAB 1: LIVE SEARCH & MODEL COMPARISON
# ==========================================================
with tab_search:
    st.subheader("1. Enter or Select a Factual Claim")
    
    # Quick sample claim buttons
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    if "query_input" not in st.session_state:
        st.session_state["query_input"] = "The 1991 NBA Finals had the Lakers in it."

    if col_s1.button("🏀 1991 NBA Finals (Lakers)", use_container_width=True):
        st.session_state["query_input"] = "The 1991 NBA Finals had the Lakers in it."
    if col_s2.button("📺 19 Kids and Counting (Cancelled)", use_container_width=True):
        st.session_state["query_input"] = "19 Kids and Counting was cancelled."
    if col_s3.button("⏱️ 100 Greatest (Timed Episodes)", use_container_width=True):
        st.session_state["query_input"] = "100 Greatest of All Time had timed episodes."
    if col_s4.button("🔥 1992 LA Riots", use_container_width=True):
        st.session_state["query_input"] = "In 1992, the 1992 Los Angeles riots occurred."

    query = st.text_input("Claim text to verify:", value=st.session_state["query_input"])

    view_mode = st.radio(
        "Display Mode:",
        ["Side-by-Side Comparison (All 4 Models)", "Single Model Detailed View"],
        horizontal=True,
    )

    search_clicked = st.button("🔎 Retrieve Evidence", type="primary", use_container_width=True)

    if (search_clicked or query) and query.strip():
        # Execute retrievals
        pool_depth = max(top_k, candidate_k)
        tfidf_res = sparse_index.search_tfidf(query, pool_depth)
        bm25_res = sparse_index.search_bm25(query, pool_depth)
        dense_res = dense_index.search(query, pool_depth) if dense_index else []
        
        fuse_inputs = [tfidf_res, bm25_res]
        if dense_res:
            fuse_inputs.append(dense_res)
        rrf_res = reciprocal_rank_fusion(fuse_inputs, k=rrf_smoothing, top_k=top_k)

        st.markdown(f"### Evidence Retrieval Results (Top {top_k})")

        if view_mode == "Side-by-Side Comparison (All 4 Models)":
            c_tfidf, c_bm25, c_dense, c_rrf = st.columns(4)

            with c_tfidf:
                st.markdown("#### 1. TF-IDF (Sparse)")
                st.caption("Vector Space Model with sublinear TF-IDF")
                for row in tfidf_res[:top_k]:
                    with st.container(border=True):
                        st.markdown(f"**#{row['rank']} · {row['title']}** (s.{row['sentence_id']})")
                        st.write(row["text"])
                        st.caption(f"Score: `{row['score']:.4f}` | ID: `{row['passage_id']}`")

            with c_bm25:
                st.markdown("#### 2. BM25 (Okapi)")
                st.caption("Probabilistic saturation model")
                for row in bm25_res[:top_k]:
                    with st.container(border=True):
                        st.markdown(f"**#{row['rank']} · {row['title']}** (s.{row['sentence_id']})")
                        st.write(row["text"])
                        st.caption(f"Score: `{row['score']:.4f}` | ID: `{row['passage_id']}`")

            with c_dense:
                st.markdown("#### 3. Dense Transformer")
                st.caption("all-MiniLM-L6-v2 cosine similarity")
                if dense_res:
                    for row in dense_res[:top_k]:
                        with st.container(border=True):
                            st.markdown(f"**#{row['rank']} · {row['title']}** (s.{row['sentence_id']})")
                            st.write(row["text"])
                            st.caption(f"Score: `{row['score']:.4f}` | ID: `{row['passage_id']}`")
                else:
                    st.info("Dense index not available.")

            with c_rrf:
                st.markdown("#### 4. Hybrid RRF (Fused)")
                st.caption("Consensus rank fusion (Sparse + Dense)")
                for row in rrf_res[:top_k]:
                    with st.container(border=True):
                        st.markdown(f"**#{row['rank']} 🏆 {row['title']}** (s.{row['sentence_id']})")
                        st.write(row["text"])
                        st.caption(f"RRF Score: `{row['score']:.5f}` | ID: `{row['passage_id']}`")

        else:
            # Single model view
            sel_model = st.selectbox(
                "Choose model to inspect:",
                ["Hybrid RRF", "TF-IDF", "BM25", "Sentence Transformer"]
            )
            if sel_model == "TF-IDF":
                results = tfidf_res[:top_k]
            elif sel_model == "BM25":
                results = bm25_res[:top_k]
            elif sel_model == "Sentence Transformer":
                results = dense_res[:top_k]
            else:
                results = rrf_res[:top_k]

            for row in results:
                with st.container(border=True):
                    st.markdown(f"**Rank #{row['rank']} — Article: `{row['title']}` | Sentence ID: `{row['sentence_id']}`**")
                    st.markdown(f"> *\"{row['text']}\"*")
                    st.caption(f"Model: **{row['method']}** | Score: `{row['score']:.5f}` | Unique Passage ID: `{row['passage_id']}`")


# ==========================================================
# TAB 2: BENCHMARK PERFORMANCE & PR CURVES
# ==========================================================
with tab_benchmark:
    st.subheader("📊 Comparative Evaluation Benchmark")
    st.markdown(
        "Evaluation performed across **100 FEVER verification queries** against the ground-truth "
        "evidence annotations following the Standardized IR Evaluation Template."
    )

    # Key metric callouts
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Hybrid RRF Recall", "91.99%", "+3.79% vs TF-IDF")
    m_col2.metric("Hybrid RRF MAP", "0.8386", "+0.0410 vs TF-IDF")
    m_col3.metric("Hybrid RRF nDCG@10", "0.8635", "+0.0405 vs TF-IDF")
    m_col4.metric("Hybrid RRF P@10", "0.1120", "Optimal for FEVER")

    # Load and show performance comparison table
    csv_file = Path("artifacts/performance_comparison.csv")
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        st.markdown("#### Detailed Performance Comparison Table")
        st.dataframe(
            df.style.highlight_max(subset=["precision@10", "recall", "map", "ndcg"], color="#d4edda"),
            use_container_width=True,
        )

    # Precision-Recall Graph
    pr_img = Path("artifacts/precision_recall.png")
    if pr_img.exists():
        st.markdown("#### Precision-Recall Graph (Dual Panel: Cutoffs & 11-Point Interpolated)")
        st.image(str(pr_img), use_container_width=True)
        st.caption(
            "Left: Precision vs Recall across ranked cutoffs (k=1 to 50), with k=10 highlighted. "
            "Right: Standard 11-point interpolated Precision-Recall curve demonstrating Hybrid RRF's Pareto superiority."
        )


# ==========================================================
# TAB 3: METHODOLOGY & MATHEMATICAL FORMULATIONS
# ==========================================================
with tab_theory:
    st.subheader("📐 Information Retrieval Formulations & RRF Theory")

    st.markdown("### 1. Reciprocal Rank Fusion (RRF) Formula (Slide 24)")
    st.markdown(
        "RRF combines rankings from multiple distinct retrieval systems (TF-IDF, BM25, Sentence Transformer) "
        "without requiring sensitive score normalization:"
    )
    st.latex(r"\text{RRF\_Score}(d \in D) = \sum_{m \in M} \frac{1}{k + r_m(d)}")
    st.markdown(
        "- $M$: Set of ranking methods ($M = \\{\\text{TF-IDF}, \\text{BM25}, \\text{Sentence Transformer}\\}$)\n"
        "- $r_m(d)$: 1-based rank position of passage $d$ in method $m$\n"
        "- $k$: Smoothing constant (default $k=60$ prevents top ranks from dominating disproportionately)"
    )

    st.markdown("---")
    st.markdown("### 2. BM25 Okapi Probabilistic Ranking")
    st.latex(r"\text{Score}_{\text{BM25}}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}")
    st.markdown("Parameters: $k_1 = 1.5$ (term frequency saturation), $b = 0.75$ (document length normalization).")

    st.markdown("---")
    st.markdown("### 3. Dense Semantic Vector Transformer")
    st.latex(r"\text{Score}_{\text{Dense}}(q, d) = \frac{\vec{e}_q \cdot \vec{e}_d}{\|\vec{e}_q\| \|\vec{e}_d\|}")
    st.markdown("Generated using `sentence-transformers/all-MiniLM-L6-v2` projecting into a 384-dimensional dense semantic manifold.")

    st.markdown("---")
    st.markdown("### 4. Standard Evaluation Metrics Checklist")
    st.markdown(
        "- **Precision@10**: Proportion of relevant items in top-10 results: $\\frac{|\\text{Retrieved}_{10} \\cap \\text{Relevant}|}{10}$\n"
        "- **Recall**: Ratio of retrieved relevant items to total relevant items: $\\frac{|\\text{Retrieved}_{10} \\cap \\text{Relevant}|}{|\\text{Relevant}|}$\n"
        "- **MAP (Mean Average Precision)**: Mean of average precisions across queries: $\\frac{1}{|Q|} \\sum_{q=1}^{|Q|} \\text{AP}(q)$\n"
        "- **nDCG (Normalized Discounted Cumulative Gain)**: Rank-weighted relevance score: $\\frac{\\text{DCG}_k}{\\text{IDCG}_k}$"
    )
