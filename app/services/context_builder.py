def build_context(chunks: list[dict]) -> str:
    if not chunks:
        return ""
    chunk_lookup: dict[tuple[str, int], dict] = {
        (c["doc_id"], c["chunk_index"]): c
        for c in chunks
    }
    visited: set[tuple[str, int]] = set()
    parts:   list[str]            = []
    for chunk in chunks:
        key = (chunk["doc_id"], chunk["chunk_index"])
        if key in visited:
            continue
        visited.add(key)
        merged_text   = chunk["text"]
        current_index = chunk["chunk_index"]
        while True:
            next_key = (chunk["doc_id"], current_index + 1)
            if next_key in chunk_lookup and next_key not in visited:
                visited.add(next_key)
                merged_text   += "\n" + chunk_lookup[next_key]["text"]
                current_index += 1
            else:
                break
        section = chunk.get("section_title") or "—"
        header  = f"[Doc: {chunk['doc_id']} | Section: {section}]"
        parts.append(f"{header}\n{merged_text}")
    return "\n\n".join(parts)