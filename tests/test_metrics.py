from fever_ir.evaluation.metrics import precision_at_k, recall_at_k, average_precision, ndcg_at_k

def test_metrics():
    retrieved = ["a", "x", "b", "y"]
    relevant = {"a", "b"}
    assert precision_at_k(retrieved, relevant, 4) == 0.5
    assert recall_at_k(retrieved, relevant, 4) == 1.0
    assert average_precision(retrieved, relevant) > 0.7
    assert ndcg_at_k(retrieved, relevant, 4) > 0.8
