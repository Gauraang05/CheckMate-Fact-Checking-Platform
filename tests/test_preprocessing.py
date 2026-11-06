from fever_ir.retrieval.preprocessing import clean_text, preprocess_text, preprocess_for_tfidf


def test_clean_text_removes_fever_markers():
    text = "Roman Atwood -LRB- born May 28 , 1983 -RRB- is an American YouTube personality ."
    cleaned = clean_text(text)
    assert "-LRB-" not in cleaned
    assert "-RRB-" not in cleaned
    assert "(" in cleaned and ")" in cleaned


def test_preprocess_tokenization_and_stemming():
    text = "The investigator verified multiple running applications."
    tokens = preprocess_text(text, stem=True, remove_stopwords=True)
    # Stop words like 'the' are removed, 'verified' -> 'verifi', 'running' -> 'run', 'applications' -> 'applic'
    assert "the" not in tokens
    assert "verifi" in tokens
    assert "run" in tokens
    assert "applic" in tokens


def test_preprocess_for_tfidf():
    text = "Document verification system"
    result = preprocess_for_tfidf(text)
    assert isinstance(result, str)
    assert "verif" in result or "system" in result

