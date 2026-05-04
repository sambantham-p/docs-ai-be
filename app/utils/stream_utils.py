# Split large text into smaller chunks for streaming.
def split_text(text: str, max_len: int = 200):
    for i in range(0, len(text), max_len):
        yield text[i:i + max_len]