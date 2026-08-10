"""Central paths and settings. Import this rather than hardcoding paths anywhere."""

from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"
PDFS = DATA / "pdfs"          # fetched-once FIA documents
CACHE = DATA / "cache"        # FastF1 session cache
INDEX = DATA / "index"        # embeddings, BM25, DuckDB file
DB = INDEX / "corpus.duckdb"

for _p in (PDFS, CACHE, INDEX):
    _p.mkdir(parents=True, exist_ok=True)

# --- Corpus scope -----------------------------------------------------------
# Two seasons gives real precedent depth without an hours-long first scrape.
SEASONS = [2025, 2026]

FIA_BASE = "https://www.fia.com"
FIA_F1_DOCS = f"{FIA_BASE}/documents/championships/fia-formula-one-world-championship-14"

# Politeness: the FIA is a free public source and we fetch each PDF exactly once.
FETCH_DELAY_SECONDS = 1.0
USER_AGENT = "f1-race-control/0.1 (personal learning project)"

# --- Models -----------------------------------------------------------------
# Both run locally in Ollama. Zero cost, no API key.
OLLAMA_HOST = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768
LLM_MODEL = "qwen3:4b"

# nomic-embed-text is trained with asymmetric task prefixes. Omitting them puts the
# model off-distribution and measurably corrupts ranking: without these, two driving
# offences scored LESS similar to each other (0.457) than one did to an unrelated pit
# lane speed rule (0.524). With them, the ordering corrects (0.640 vs 0.541).
# See docs/concepts/01-embeddings.md. Never embed raw text.
EMBED_PREFIX_DOC = "search_document: "
EMBED_PREFIX_QUERY = "search_query: "

# --- Retrieval --------------------------------------------------------------
TOP_K = 8            # passages handed to the model
RRF_K = 60           # reciprocal rank fusion constant (explained in docs/concepts/)
