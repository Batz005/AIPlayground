"""
Tests for vectorizer.py
"""

import pytest
from utils import vectorizer

def test_vectorize_string():
    text = "This is a test sentence."
    vec = vectorizer.vectorize_string(text)
    assert isinstance(vec.tolist(), list)  # should be convertible to list
    assert len(vec) > 0  # embedding has dimensions

def test_vectorize_all():
    texts = ["First sentence.", "Second sentence."]
    vecs = vectorizer.vectorize_all(texts)
    assert len(vecs) == len(texts)  # one embedding per input
    assert all(len(v) == len(vecs[0]) for v in vecs)  # consistent dimension sizes
