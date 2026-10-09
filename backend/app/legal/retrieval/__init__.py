"""Partitioned retrieval for EchoMind Law (S1.6, SEC-03; decision D-005).

Every matter has its own dense (FAISS) and sparse (BM25) index. Searching needs an
AuthorizedScope, which only the policy engine can mint and which is re-checked on every use.
"""
