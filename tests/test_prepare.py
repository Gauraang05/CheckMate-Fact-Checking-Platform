from fever_ir.data.prepare import normalize

def test_normalize():
    assert normalize("  A\u2019s   claim  ") == "A's claim"
