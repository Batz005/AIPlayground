from modules.retrievers.factory import RetrieverFactory

def test_semantic_retrieve_awards():
    chunks = [
        "Bharath got an award for his work at Samsung.",
        "He enjoys solving algorithms and competitive programming.",
        "Today the weather is sunny and bright."
    ]
    retriever = RetrieverFactory.create({"type": "semantic"})
    res = retriever.run("What awards did Bharath get?", chunks, top_k=1)
    assert "award" in res[0][0].lower()