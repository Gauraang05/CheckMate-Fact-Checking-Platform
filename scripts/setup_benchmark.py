import json
import os
import zipfile
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TORCH", "1")

from fever_ir.retrieval.core import EvidenceIndex, DenseIndex
from fever_ir.evaluation.runner import evaluate_retrievers, save_report


def setup_benchmark_and_evaluate(
    wiki_zip_path="data/raw/wiki-pages.zip",
    train_path="data/raw/train.jsonl",
    output_dir="artifacts",
    max_passages=5000,
    max_eval_claims=100,
):
    print("Step 1: Extracting articles from wiki-001...")
    articles = {}
    with zipfile.ZipFile(wiki_zip_path) as z:
        with z.open("wiki-pages/wiki-001.jsonl") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    page = json.loads(line)
                    t = page.get("id") or page.get("title")
                    if t:
                        lines = page.get("lines", "").splitlines()
                        sents = {}
                        for l in lines:
                            parts = l.split("\t", 1)
                            if len(parts) == 2 and parts[1].strip():
                                sents[parts[0]] = parts[1].strip()
                        if sents:
                            articles[t] = sents
                except Exception:
                    pass

    print(f"Loaded {len(articles)} Wikipedia articles.")

    print("Step 2: Identifying valid FEVER claims whose evidence is in the corpus...")
    matched_claims = []
    with open(train_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ev_items = []
            valid = True
            has_ev = False
            for g in r.get("evidence", []) or []:
                for it in g or []:
                    if len(it) >= 4 and it[2] is not None and it[3] is not None:
                        has_ev = True
                        t = it[2]
                        s = str(it[3])
                        if t in articles and s in articles[t]:
                            ev_items.append((t, s))
                        else:
                            valid = False
            if has_ev and valid and ev_items:
                matched_claims.append(r)
                if len(matched_claims) >= max_eval_claims:
                    break

    print(f"Selected {len(matched_claims)} FEVER claims with verified evidence.")

    print("Step 3: Creating passage collection (true evidence + distractors)...")
    passages = []
    seen_ids = set()

    # 1. Guarantee all true evidence passages are included
    for claim in matched_claims:
        for g in claim.get("evidence", []) or []:
            for it in g or []:
                if len(it) >= 4 and it[2] is not None and it[3] is not None:
                    t = it[2]
                    s = str(it[3])
                    pid = f"{t}::{s}"
                    if pid not in seen_ids and t in articles and s in articles[t]:
                        passages.append({
                            "passage_id": pid,
                            "title": t,
                            "sentence_id": s,
                            "text": articles[t][s],
                        })
                        seen_ids.add(pid)

    print(f"Added {len(passages)} ground-truth evidence passages.")

    # 2. Add full articles of the matched titles (in-document distractors)
    for claim in matched_claims:
        for g in claim.get("evidence", []) or []:
            for it in g or []:
                if len(it) >= 4 and it[2] is not None and it[2] in articles:
                    t = it[2]
                    for s, text in articles[t].items():
                        pid = f"{t}::{s}"
                        if pid not in seen_ids:
                            passages.append({
                                "passage_id": pid,
                                "title": t,
                                "sentence_id": s,
                                "text": text,
                            })
                            seen_ids.add(pid)

    # 3. Add random distractor passages from other Wikipedia articles up to max_passages
    for t, sents in articles.items():
        if len(passages) >= max_passages:
            break
        for s, text in sents.items():
            pid = f"{t}::{s}"
            if pid not in seen_ids:
                passages.append({
                    "passage_id": pid,
                    "title": t,
                    "sentence_id": s,
                    "text": text,
                })
                seen_ids.add(pid)
                if len(passages) >= max_passages:
                    break

    print(f"Total passages indexed: {len(passages):,} (True evidence + distractors).")

    # Save passages and claims for reproducibility
    proc_dir = Path("data/processed")
    proc_dir.mkdir(parents=True, exist_ok=True)
    with open(proc_dir / "eval_passages.jsonl", "w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(proc_dir / "eval_claims.jsonl", "w", encoding="utf-8") as f:
        for c in matched_claims:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # Step 4: Build Sparse Index (TF-IDF + BM25 with Stemming)
    print("Step 4: Building sparse index (TF-IDF + BM25 with stemming)...")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sparse_index = EvidenceIndex(passages)
    sparse_index.save_sparse(out / "sparse_index.pkl")

    # Step 5: Build Dense Index (SentenceTransformer)
    print("Step 5: Building dense index (Sentence Transformer: all-MiniLM-L6-v2)...")
    dense_index = DenseIndex(passages, model_name="sentence-transformers/all-MiniLM-L6-v2")
    dense_index.save(out / "dense_index.npz")

    # Step 6: Run Comprehensive IR Evaluation across all 4 models
    print("Step 6: Running evaluation across all 4 ranking models...")
    report, pr_data = evaluate_retrievers(
        claims=matched_claims,
        index=sparse_index,
        dense=dense_index,
        top_k=10,
        candidate_k=50,
    )

    save_report(report, pr_data, out)
    print("\n================ BENCHMARK RESULTS ================")
    print(json.dumps(report, indent=2))
    print(f"Artifacts successfully saved to {out.resolve()}")


if __name__ == "__main__":
    setup_benchmark_and_evaluate()
