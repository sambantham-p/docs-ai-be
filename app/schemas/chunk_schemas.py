from dataclasses import dataclass, asdict
from typing import Optional, TypedDict


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_index: int
    text: str
    section: str
    section_index: int
    sentence_start: int
    sentence_end: int
    char_start: int
    char_end: int
    token_count: int
    chunk_hash: str

    def to_dict(self):
        return asdict(self)

    def validate(self, doc_len: int):
        if not self.text.strip():
            raise ValueError(f"Chunk {self.chunk_index}: empty text")
        if self.token_count <= 0:
            raise ValueError(f"Chunk {self.chunk_index}: zero token count")
        if not (0 <= self.char_start <= self.char_end <= doc_len):
            raise ValueError(
                f"Chunk {self.chunk_index}: invalid char range "
                f"[{self.char_start}, {self.char_end}] for doc_len={doc_len}"
            )
        if self.sentence_start > self.sentence_end:
            raise ValueError(
                f"Chunk {self.chunk_index}: sentence_start ({self.sentence_start}) "
                f"> sentence_end ({self.sentence_end})"
            )


class TokenChunk(TypedDict):
    text: str
    token_start: int
    token_end: int
    chunk_id: int
    doc_id: Optional[str]
