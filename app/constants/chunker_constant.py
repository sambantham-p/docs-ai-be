import re

DEFAULT_CHUNK_MAX_TOKENS = 512
DEFAULT_OVERLAP_SENTENCES = 2
DEFAULT_VALIDATE_CHUNKS = True
TOKEN_ENCODING_NAME = "cl100k_base"
SPACY_LANGUAGE = "en"
MAX_SENTENCE_TOKENS = 200
MARKDOWN_HEADING_RE = re.compile(r"^#{1,6}\s+\S.*$", re.MULTILINE)
NUMBERED_HEADING_RE = re.compile(r"^\d+(?:\.\d+)*\.?\s+")
CAPS_HEADING_RE = re.compile(r"^[A-Z][A-Z0-9\s\-:]{1,}[A-Z0-9]$")
HEADING_WORD_STRIP_CHARS = ".,:;!()-"
ABBREV_STRIP_CHARS = ".,:;!?)("
ABBREVS = {
    "Mr", "Ms", "Mrs", "Dr", "Prof", "Sr", "Jr", "St", "vs", "etc", "U.S",
    "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    "No", "Vol", "pp", "ed", "Fig", "Dept", "Corp", "Inc", "Ltd", "Ave",
}
BLACKLIST = {"WARNING", "NOTE", "IMPORTANT"}
