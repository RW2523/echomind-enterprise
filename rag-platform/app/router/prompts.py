"""Prompts for RAG answer generation and evidence formatting."""

SYSTEM_RAG = """You are a helpful assistant. Answer based only on the provided context. If the context does not contain enough information, say so clearly. Include citations: for transcripts cite (transcript_id, time range, location); for documents cite (doc_title, page/section)."""

USER_RAG_TEMPLATE = """Context:
{context}

Question: {question}

Answer with citations."""

EVIDENCE_BLOCK_HEADER = "Retrieved Evidence (validation):"
EVIDENCE_BLOCK_FOOTER = "--- End Evidence ---"

def build_rag_prompt(context: str, question: str) -> list:
    return [
        {"role": "system", "content": SYSTEM_RAG},
        {"role": "user", "content": USER_RAG_TEMPLATE.format(context=context, question=question)},
    ]

def format_evidence_block(evidence: list) -> str:
    """Format evidence list for validation-friendly response block."""
    lines = [EVIDENCE_BLOCK_HEADER]
    for e in evidence:
        lines.append(str(e))
    lines.append(EVIDENCE_BLOCK_FOOTER)
    return "\n".join(lines)
