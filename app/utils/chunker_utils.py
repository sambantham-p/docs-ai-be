import hashlib
from typing import Optional, TypedDict

import tiktoken

from app.constants.chunker_constant import TOKEN_ENCODING_NAME

_ENCODER: tiktoken.Encoding | None = None

class TokenChunk(TypedDict):
    text: str
    token_start: int
    token_end: int
    chunk_id: int
    doc_id: Optional[str]

def get_encoder() -> tiktoken.Encoding:
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = tiktoken.get_encoding(TOKEN_ENCODING_NAME)
    return _ENCODER


def warmup_tokenizer() -> None:
    get_encoder()

def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def count_tokens(text: str) -> int:
    return len(encode_tokens(text))


def encode_tokens(text: str) -> list[int]:
    return get_encoder().encode(text, disallowed_special=())


def decode_tokens(token_ids: list[int]) -> str:
    return get_encoder().decode_bytes(token_ids).decode("utf-8", errors="replace")

def split_by_tokens(
    text: str,
    max_tokens: int,
    overlap: int = 0,
    doc_id: Optional[str] = None,
) -> list[TokenChunk]:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text.strip():
        return []
    if max_tokens <= 0:
        raise ValueError("max_tokens must be > 0")
    if not (0 <= overlap < max_tokens):
        raise ValueError("overlap must be in range [0, max_tokens)")

    tokens = encode_tokens(text)
    total_tokens = len(tokens)
    if total_tokens <= max_tokens:
        return [
            {
                "text": decode_tokens(tokens),
                "token_start": 0,
                "token_end": total_tokens,
                "chunk_id": 0,
                "doc_id": doc_id,
            }
        ]

    chunks: list[TokenChunk] = []
    step_size = max_tokens - overlap
    for chunk_index, start in enumerate(range(0, total_tokens, step_size)):
        end = min(start + max_tokens, total_tokens)
        chunks.append({
            "text": decode_tokens(tokens[start:end]),
            "token_start": start,
            "token_end": end,
            "chunk_id": chunk_index,
            "doc_id": doc_id,
        })
        if end == total_tokens:
            break
    return chunks