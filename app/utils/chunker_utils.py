import hashlib

import tiktoken

from app.constants.chunker_constant import ABBREVS, ABBREV_STRIP_CHARS, TOKEN_ENCODING_NAME

_ENCODER: tiktoken.Encoding | None = None

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
