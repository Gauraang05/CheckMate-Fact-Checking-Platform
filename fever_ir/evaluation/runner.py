from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import matplotlib.pyplot as plt
import numpy as np

from .metrics import evaluate_query, precision_at_k, recall_at_k, interpolated_precision_recall
from ..retrieval.core import reciprocal_rank_fusion


def relevant_ids(row: dict) -> set[str]:
    """Extract evidence set in format 'Page_Title::Sentence_ID' from FEVER claim row."""
    ids = set()
    for group in row.get("evidence", []) or []:
        for item in group or []:
            if len(item) >= 4 and item[2] is not None and item[3] is not None:
                ids.add(f"{item[2]}::{item[3]}")
    return ids


def evaluate_retrievers(
    claims: List[dict],
    index,
    dense=None,
    max_queries: Optional[int] = None,
    top_k: int = 10,
    candidate_k: int = 50,
) -> tuple[List[dict], dict]:
    """
    Evaluates TF-IDF, BM25, Sentence Transformer (if available), and Hybrid RRF.
    Returns:
        (report_table, pr_curve_data)
    """
    method_names = ["TF-IDF", "BM25"]
    if dense is not None:
        method_names.append("Sentence Transformer")
    method_names.append("Hybrid RRF")

    totals = {m: {"precision@10": 0.0, "recall": 0.0, "map": 0.0, "ndcg": 0.0} for m in method_names}
    
    # Track PR curve data across cutoffs and 11-point interpolated recall
    cutoffs = [1, 2, 3, 5, 10, 15, 20, 30, 50]
    cutoff_pr = {m: {k: {"precision": 0.0, "recall": 0.0} for k in cutoffs} for m in method_names}
    interp_pr = {m: np.zeros(11) for m in method_names}
    
    used = 0
    eval_claims = claims[:max_queries] if max_queries else claims
    
    for row in eval_claims:
        relevant = relevant_ids(row)
        if not relevant:
            continue
            
        claim_text = row["claim"]
        
        # Candidate generation
        tfidf_candidates = index.search_tfidf(claim_text, max(candidate_k, top_k))
        bm25_candidates = index.search_bm25(claim_text, max(candidate_k, top_k))
        dense_candidates = dense.search(claim_text, max(candidate_k, top_k)) if dense is not None else []
        
        # Hybrid RRF fusion
        fuse_inputs = [tfidf_candidates, bm25_candidates]
        if dense is not None and dense_candidates:
            fuse_inputs.append(dense_candidates)
            
        rrf_results = reciprocal_rank_fusion(fuse_inputs, k=60, top_k=max(candidate_k, top_k))
        
        method_results = {
            "TF-IDF": tfidf_candidates,
            "BM25": bm25_candidates,
            "Hybrid RRF": rrf_results,
        }
        if dense is not None:
            method_results["Sentence Transformer"] = dense_candidates
            
        for method, results in method_results.items():
            retrieved_ids = [x["passage_id"] for x in results]
            
            # Standard metrics at top_k
            q_metrics = evaluate_query(retrieved_ids, relevant, k=top_k)
            for m_key, val in q_metrics.items():
                totals[method][m_key] += val
                
            # Cutoff metrics for PR curve
            for k_val in cutoffs:
                cutoff_pr[method][k_val]["precision"] += precision_at_k(retrieved_ids, relevant, k_val)
                cutoff_pr[method][k_val]["recall"] += recall_at_k(retrieved_ids, relevant, k_val)
                
            # 11-point interpolated PR
            interp_curve = interpolated_precision_recall(retrieved_ids, relevant)
            interp_pr[method] += np.array(interp_curve)
            
        used += 1

    report = []
    # Guarantee consistent order: TF-IDF, BM25, Sentence Transformer, Hybrid RRF
    ordered_methods = [m for m in ["TF-IDF", "BM25", "Sentence Transformer", "Hybrid RRF"] if m in totals]
    
    for method in ordered_methods:
        values = totals[method]
        report.append({
            "method": method,
            "precision@10": round(values["precision@10"] / used if used else 0.0, 4),
            "recall": round(values["recall"] / used if used else 0.0, 4),
            "map": round(values["map"] / used if used else 0.0, 4),
            "ndcg": round(values["ndcg"] / used if used else 0.0, 4),
            "queries": used,
        })
        
    pr_curve_data = {
        "cutoffs": cutoffs,
        "cutoff_pr": {
            m: {
                k: {
                    "precision": round(cutoff_pr[m][k]["precision"] / used, 4) if used else 0.0,
                    "recall": round(cutoff_pr[m][k]["recall"] / used, 4) if used else 0.0,
                }
                for k in cutoffs
            }
            for m in ordered_methods
        },
        "interpolated_11point": {
            m: [round(float(x), 4) for x in (interp_pr[m] / used if used else np.zeros(11))]
            for m in ordered_methods
        },
    }
    
    return report, pr_curve_data


def save_report(report: List[dict], pr_data: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    csv_path = output_dir / "performance_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=report[0].keys())
        writer.writeheader()
        writer.writerows(report)
        
    # Save JSON
    json_path = output_dir / "performance_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": report, "pr_curves": pr_data}, f, indent=2)
        
    # Generate Dual-Panel Precision-Recall Graph
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    color_map = {
        "TF-IDF": "#1f77b4",            # Blue
        "BM25": "#ff7f0e",             # Orange
        "Sentence Transformer": "#2ca02c", # Green
        "Hybrid RRF": "#d62728",       # Crimson Red
    }
    marker_map = {
        "TF-IDF": "o",
        "BM25": "s",
        "Sentence Transformer": "^",
        "Hybrid RRF": "D",
    }
    
    # Subplot 1: Precision vs Recall at rank cutoffs (k=1..50)
    for row in report:
        method = row["method"]
        pts = pr_data["cutoff_pr"][method]
        recalls = [pts[k]["recall"] for k in pr_data["cutoffs"]]
        precisions = [pts[k]["precision"] for k in pr_data["cutoffs"]]
        color = color_map.get(method, "gray")
        marker = marker_map.get(method, "o")
        ax1.plot(recalls, precisions, label=method, color=color, marker=marker, linewidth=2, markersize=5)
        # Highlight k=10 operating point
        k10_rec = pts[10]["recall"]
        k10_prec = pts[10]["precision"]
        ax1.scatter([k10_rec], [k10_prec], color=color, s=90, edgecolors="black", zorder=5)
        ax1.annotate(f"{method} (k=10)", (k10_rec, k10_prec), xytext=(5, 5), textcoords="offset points", fontsize=8)
        
    ax1.set_xlabel("Recall", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Precision", fontsize=11, fontweight="bold")
    ax1.set_title("Precision vs Recall (Rank Cutoffs k=1 to 50)", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.set_ylim(-0.02, 1.02)
    
    # Subplot 2: 11-Point Interpolated Precision-Recall Curve
    recall_levels = [i / 10.0 for i in range(11)]
    for row in report:
        method = row["method"]
        interp_p = pr_data["interpolated_11point"][method]
        color = color_map.get(method, "gray")
        marker = marker_map.get(method, "o")
        ax2.plot(recall_levels, interp_p, label=f"{method} (MAP: {row['map']:.3f})", color=color, marker=marker, linewidth=2, markersize=5)
        
    ax2.set_xlabel("Standard Recall Level", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Interpolated Precision", fontsize=11, fontweight="bold")
    ax2.set_title("11-Point Interpolated Precision-Recall Curve", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.set_ylim(-0.02, 1.02)
    
    plt.suptitle("FEVER Evidence Retrieval: Comparative Evaluation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "precision_recall.png", dpi=200)
    plt.close()
