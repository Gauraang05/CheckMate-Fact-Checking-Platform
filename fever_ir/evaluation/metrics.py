import math
from typing import List, Set, Dict


def precision_at_k(retrieved: List[str], relevant: Set[str], k: int = 10) -> float:
    """Proportion of retrieved items in top-k that are relevant."""
    if k <= 0:
        return 0.0
    hits = sum(1 for x in retrieved[:k] if x in relevant)
    return hits / k


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int = 10) -> float:
    """Ratio of retrieved relevant items in top-k to total relevant items."""
    if not relevant:
        return 0.0
    hits = sum(1 for x in retrieved[:k] if x in relevant)
    return hits / len(relevant)


def average_precision(retrieved: List[str], relevant: Set[str]) -> float:
    """Average precision across all relevant item positions."""
    if not relevant:
        return 0.0
    hits = 0
    total = 0.0
    for i, item in enumerate(retrieved, start=1):
        if item in relevant:
            hits += 1
            total += hits / i
    return total / len(relevant)


def ndcg_at_k(retrieved: List[str], relevant: Set[str], k: int = 10) -> float:
    """Normalized Discounted Cumulative Gain at rank k."""
    if not relevant or k <= 0:
        return 0.0
    dcg = sum((1.0 / math.log2(i + 2)) for i, item in enumerate(retrieved[:k]) if item in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def interpolated_precision_recall(retrieved: List[str], relevant: Set[str]) -> List[float]:
    """
    Computes standard 11-point interpolated precision at recall levels [0.0, 0.1, ..., 1.0].
    P_interp(r) = max_{r' >= r} P(r')
    """
    if not relevant:
        return [0.0] * 11
        
    precisions = []
    recalls = []
    hits = 0
    for i, item in enumerate(retrieved, start=1):
        if item in relevant:
            hits += 1
            precisions.append(hits / i)
            recalls.append(hits / len(relevant))
            
    if not precisions:
        return [0.0] * 11
        
    recall_levels = [i / 10.0 for i in range(11)]
    interp_p = []
    for r in recall_levels:
        eligible = [p for p, rec in zip(precisions, recalls) if rec >= r]
        interp_p.append(max(eligible) if eligible else 0.0)
    return interp_p


def evaluate_query(retrieved: List[str], relevant: Set[str], k: int = 10) -> Dict[str, float]:
    return {
        "precision@10": precision_at_k(retrieved, relevant, k),
        "recall": recall_at_k(retrieved, relevant, k),
        "map": average_precision(retrieved, relevant),
        "ndcg": ndcg_at_k(retrieved, relevant, k),
    }
