from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

from .data.download import run as download_run
from .data.prepare import run as prepare_run
from .retrieval.core import EvidenceIndex, DenseIndex, load_passages
from .evaluation.metrics import evaluate_query
from .evaluation.runner import evaluate_retrievers, save_report


def build_index(args):
    passages_file = args.passages_file or "passages.jsonl"
    passages_path = Path(args.processed_dir) / passages_file
    if not passages_path.exists():
        # Check raw or fallback
        if Path(passages_file).exists():
            passages_path = Path(passages_file)
        else:
            raise SystemExit(f"Passages file {passages_path} not found.")
            
    passages = load_passages(passages_path, args.max_passages)
    out = Path(args.artifacts_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    print(f"Building sparse index (TF-IDF & BM25 with stemming) for {len(passages):,} passages...")
    sparse = EvidenceIndex(passages)
    sparse.save_sparse(out / "sparse_index.pkl")
    
    if not args.skip_dense:
        print(f"Building dense index ({args.model}) for {len(passages):,} passages...")
        dense = DenseIndex(passages, args.model)
        dense.save(out / "dense_index.npz")
    print(f"Index build complete. Saved in {out}")


def evaluate(args):
    processed = Path(args.processed_dir)
    claims_path = processed / f"{args.split}.jsonl"
    if not claims_path.exists():
        if Path(f"{args.split}.jsonl").exists():
            claims_path = Path(f"{args.split}.jsonl")
        else:
            raise SystemExit(f"Claims file {claims_path} not found.")
            
    with open(claims_path, encoding="utf-8") as f:
        claims = [json.loads(x) for x in f]
        
    sparse_path = Path(args.artifacts_dir) / "sparse_index.pkl"
    if not sparse_path.exists():
        raise SystemExit(f"Sparse index not found at {sparse_path}. Run build-index first.")
        
    index = EvidenceIndex.load_sparse(sparse_path)
    
    dense = None
    meta_path = Path(args.artifacts_dir) / "dense_index.npz.meta.json"
    if not args.skip_dense and meta_path.exists():
        try:
            print("Loading precomputed dense index...")
            dense = DenseIndex.load(Path(args.artifacts_dir) / "dense_index.npz")
        except Exception as exc:
            print(f"Dense evaluation skipped: {exc}")
            
    report, pr_data = evaluate_retrievers(
        claims=claims,
        index=index,
        dense=dense,
        max_queries=args.max_queries,
        top_k=args.top_k,
    )
    
    save_report(report, pr_data, Path(args.artifacts_dir))
    print("\n--- Performance Comparison Table ---")
    print(json.dumps(report, indent=2))
    print(f"\nArtifacts saved to {args.artifacts_dir}/ (performance_comparison.csv, performance_comparison.json, precision_recall.png)")


def main():
    p = argparse.ArgumentParser(prog="python -m fever_ir.cli")
    sub = p.add_subparsers(dest="command", required=True)
    
    d = sub.add_parser("download")
    d.add_argument("--raw-dir", default="data/raw")
    d.add_argument("--full", action="store_true")
    d.set_defaults(func=download_run)
    
    pr = sub.add_parser("prepare")
    pr.add_argument("--raw-dir", default="data/raw")
    pr.add_argument("--processed-dir", default="data/processed")
    pr.add_argument("--sample", type=int)
    pr.set_defaults(func=prepare_run)
    
    b = sub.add_parser("build-index")
    b.add_argument("--processed-dir", default="data/processed")
    b.add_argument("--passages-file", default=None)
    b.add_argument("--artifacts-dir", default="artifacts")
    b.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    b.add_argument("--max-passages", type=int)
    b.add_argument("--skip-dense", action="store_true")
    b.set_defaults(func=build_index)
    
    e = sub.add_parser("evaluate")
    e.add_argument("--processed-dir", default="data/processed")
    e.add_argument("--artifacts-dir", default="artifacts")
    e.add_argument("--split", default="dev")
    e.add_argument("--max-queries", type=int)
    e.add_argument("--top-k", type=int, default=10)
    e.add_argument("--skip-dense", action="store_true")
    e.set_defaults(func=evaluate)
    
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
