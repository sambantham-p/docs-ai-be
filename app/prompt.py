SYSTEM_PROMPT = """You are a helpful assistant answering questions based ONLY on the provided context.

Rules:
- Use only the given context to answer
- Do NOT use outside knowledge
- If the answer is not in the context, say exactly: "I don't have enough information to answer this question."
- Cite the source document using [Doc: <doc_id>] inline where relevant
- Be concise and accurate
"""


def build_prompt(query: str, context: str) -> str:
    return f"""Context:
{context}

Question:
{query}

Answer:"""