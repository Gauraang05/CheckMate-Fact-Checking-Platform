import tempfile
from pathlib import Path
from fever_ir.retrieval.core import EvidenceIndex, reciprocal_rank_fusion
from fever_ir.evaluation.runner import evaluate_retrievers, relevant_ids


def test_evidence_index_search():
    passages = [
        {"passage_id": "paris::0", "title": "Paris", "sentence_id": "0", "text": "The Eiffel Tower is in Paris, France."},
        {"passage_id": "london::0", "title": "London", "sentence_id": "0", "text": "Big Ben is a famous clock tower in London."},
        {"passage_id": "berlin::0", "title": "Berlin", "sentence_id": "0", "text": "Berlin is the capital of Germany."},
    ]
    index = EvidenceIndex(passages)
    
    tfidf_res = index.search_tfidf("Eiffel Tower Paris", k=2)
    assert len(tfidf_res) == 2
    assert tfidf_res[0]["passage_id"] == "paris::0"
    
    bm25_res = index.search_bm25("clock tower London", k=2)
    assert len(bm25_res) == 2
    assert bm25_res[0]["passage_id"] == "london::0"


def test_evaluate_retrievers_mock():
    passages = [
        {"passage_id": "Paris::0", "title": "Paris", "sentence_id": "0", "text": "The Eiffel Tower is in Paris, France."},
        {"passage_id": "London::0", "title": "London", "sentence_id": "0", "text": "Big Ben is a famous clock tower in London."},
    ]
    index = EvidenceIndex(passages)
    
    claims = [
        {
            "id": 1,
            "claim": "The Eiffel Tower is situated in Paris.",
            "evidence": [[[1, 1, "Paris", 0]]],
        }
    ]
    
    report, pr_data = evaluate_retrievers(claims, index, dense=None, top_k=2)
    assert len(report) >= 3  # TF-IDF, BM25, Hybrid RRF
    rrf_row = next(r for r in report if r["method"] == "Hybrid RRF")
    assert rrf_row["precision@10"] > 0.0
    assert rrf_row["recall"] == 1.0

