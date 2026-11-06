from fever_ir.retrieval.core import reciprocal_rank_fusion

def test_rrf_favors_consensus():
    a = [{"passage_id": "one", "text": "1", "rank": 1}, {"passage_id": "two", "text": "2", "rank": 2}]
    b = [{"passage_id": "two", "text": "2", "rank": 1}, {"passage_id": "three", "text": "3", "rank": 2}]
    result = reciprocal_rank_fusion([a, b], top_k=3)
    assert result[0]["passage_id"] == "two"
