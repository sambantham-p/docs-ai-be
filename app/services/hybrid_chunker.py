import spacy
from app.constants.chunker_constant import (
    BLACKLIST,
    CAPS_HEADING_RE,
    DEFAULT_CHUNK_MAX_TOKENS,
    DEFAULT_VALIDATE_CHUNKS,
    HEADING_WORD_STRIP_CHARS,
    MARKDOWN_HEADING_RE,
    NUMBERED_HEADING_RE,
    SPACY_LANGUAGE, MAX_SENTENCES_PER_SECTION, MAX_SENTENCES_PER_DOC,
)
from app.utils.chunker_utils import (
    count_tokens,
    decode_tokens,
    encode_tokens,
    hash_text,
)
from app.schemas.chunk_schemas import Chunk


NLP = spacy.blank(SPACY_LANGUAGE)
NLP.add_pipe("sentencizer")


def hybrid_chunk(
    text: str,
    doc_id: str,
    max_tokens: int = DEFAULT_CHUNK_MAX_TOKENS,
    validate: bool = DEFAULT_VALIDATE_CHUNKS,
) -> list[dict]:
    if not text.strip():
        raise ValueError("Empty document")
    if len(text) > 5_000_000:
        raise ValueError("Document too large")
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be > 0, got {max_tokens}")
    overlap_tokens = int(max_tokens * 0.2)
    sections = split_sections(text)
    chunk_results = []
    next_chunk_index = 0
    total_sentences = 0
    for section_index, (section_title, section_text, section_start, _) in enumerate(sections):
        sentences = split_sentences(section_text, section_start)
        section_label = section_title or f"section_{section_index}"
        if len(sentences) > MAX_SENTENCES_PER_SECTION:
            raise ValueError(
                f"Section '{section_label}' has too many sentences "
                f"({len(sentences)} > {MAX_SENTENCES_PER_SECTION}) — possible malformed input"
            )
        total_sentences += len(sentences)
        if total_sentences > MAX_SENTENCES_PER_DOC:
            raise ValueError(
                f"Document '{doc_id}' exceeded total sentence limit "
                f"({total_sentences} > {MAX_SENTENCES_PER_DOC}) "
                f"at section '{section_label}' — possible malformed input"
            )
        chunks = build_chunks(
            sentences,
            doc_id,
            section_title,
            section_index,
            max_tokens,
            overlap_tokens,
            next_chunk_index,
        )
        if validate:
            for chunk in chunks:
                chunk.validate(len(text))

        chunk_results.extend(chunk.to_dict() for chunk in chunks)
        next_chunk_index += len(chunks)
    return chunk_results


def split_sentences(text: str, offset: int) -> list[tuple[str, int, int]]:
    doc = NLP(text)
    return [
        (sent.text.strip(), offset + sent.start_char, offset + sent.end_char)
        for sent in doc.sents
        if sent.text.strip()
    ]


def split_sections(text: str) -> list[tuple[str, str, int, int]]:
    lines = text.splitlines(keepends=True)
    sections = []
    section_title = ""
    section_lines = []
    section_start = 0
    char_position = 0
    previous_line_blank = True

    for line in lines:
        stripped = line.strip()
        heading_words = [
            word.strip(HEADING_WORD_STRIP_CHARS).upper()
            for word in stripped.split()
        ]
        is_caps_heading = (
            CAPS_HEADING_RE.fullmatch(stripped)
            and previous_line_blank
            and 2 <= len(heading_words) <= 8
            and not stripped.rstrip().endswith(":")
            and not any(word in BLACKLIST for word in heading_words)
        )
        is_heading = (
            MARKDOWN_HEADING_RE.match(stripped)
            or NUMBERED_HEADING_RE.match(stripped)
            or is_caps_heading
        )
        if is_heading:
            section_content = "".join(section_lines)
            if section_content.strip():
                sections.append((section_title, section_content, section_start, char_position))
            section_title = stripped
            section_lines = []
            section_start = char_position + len(line)
        else:
            section_lines.append(line)

        previous_line_blank = not stripped
        char_position += len(line)

    section_content = "".join(section_lines)
    if section_content.strip():
        sections.append((section_title, section_content, section_start, char_position))

    return sections if sections else [("", text, 0, len(text))]


def _split_long_sentence(
    sentence: tuple[str, int, int],
    doc_id: str,
    section: str,
    section_index: int,
    max_tokens: int,
    chunk_index_start: int,
    sentence_idx: int,
) -> tuple[list[Chunk], int]:
    sentence_text, char_start, char_end = sentence
    token_ids = encode_tokens(sentence_text)
    chunks = []
    next_chunk_index = chunk_index_start
    for token_start in range(0, len(token_ids), max_tokens):
        sub_tokens = token_ids[token_start: token_start + max_tokens]
        sub_text = decode_tokens(sub_tokens)
        chunks.append(Chunk(
            doc_id=doc_id,
            chunk_index=next_chunk_index,
            text=sub_text,
            section=section,
            section_index=section_index,
            sentence_start=sentence_idx,
            sentence_end=sentence_idx,
            char_start=char_start,
            char_end=char_end,
            token_count=len(sub_tokens),
            chunk_hash=hash_text(sub_text),
        ))
        next_chunk_index += 1
    return chunks, next_chunk_index


def build_chunks(
    sentences: list[tuple[str, int, int]],
    doc_id: str,
    section: str,
    section_index: int,
    max_tokens: int,
    overlap_tokens: int,
    start_idx: int,
) -> list[Chunk]:
    if not sentences:
        return []
    sentence_token_counts = [count_tokens(s[0]) for s in sentences]
    chunks = []
    chunk_index = start_idx
    sentence_index = 0
    sentence_count = len(sentences)
    while sentence_index < sentence_count:
        current_slice = []
        current_token_total = 0
        window_end = sentence_index
        # Build window
        while window_end < sentence_count:
            token_count = sentence_token_counts[window_end]
            # Long sentence: split into sub-chunks, no overlap, advance
            if not current_slice and token_count > max_tokens:
                split_chunks, chunk_index = _split_long_sentence(
                    sentences[window_end],
                    doc_id, section, section_index,
                    max_tokens, chunk_index, window_end,
                )
                chunks.extend(split_chunks)
                sentence_index = window_end + 1
                window_end = sentence_index
                break
            if current_slice and current_token_total + token_count > max_tokens:
                break
            current_slice.append(sentences[window_end])
            current_token_total += token_count
            window_end += 1
        if not current_slice:
            continue
        chunk = _finalize_chunk(
            current_slice, doc_id, section, section_index,
            chunk_index, sentence_index, max_tokens,
        )
        chunks.append(chunk)
        chunk_index += 1
        # Token-based overlap backtrack (~20% of max_tokens)
        if overlap_tokens > 0 and window_end > sentence_index + 1:
            backtrack_tokens = 0
            backtrack_index = window_end - 1
            while backtrack_index > sentence_index and backtrack_tokens < overlap_tokens:
                backtrack_tokens += sentence_token_counts[backtrack_index]
                backtrack_index -= 1
            next_index = backtrack_index + 1
        else:
            next_index = window_end
        sentence_index = max(next_index, sentence_index + 1)
    return chunks


def _finalize_chunk(
    sentence_slice: list[tuple[str, int, int]],
    doc_id: str,
    section: str,
    section_index: int,
    chunk_index: int,
    sentence_start_index: int,
    max_tokens: int,
) -> Chunk:
    text = " ".join(sentence[0] for sentence in sentence_slice)
    token_ids = encode_tokens(text)
    if len(token_ids) > max_tokens:
        token_ids = token_ids[:max_tokens]
        text = decode_tokens(token_ids)

    return Chunk(
        doc_id=doc_id,
        chunk_index=chunk_index,
        text=text,
        section=section,
        section_index=section_index,
        sentence_start=sentence_start_index,
        sentence_end=sentence_start_index + len(sentence_slice) - 1,
        char_start=sentence_slice[0][1],
        char_end=sentence_slice[-1][2],
        token_count=len(token_ids),
        chunk_hash=hash_text(text),
    )