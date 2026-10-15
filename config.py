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

# Token budget for one card. It covers the model's reasoning as well as its answer,
# because Ollama draws both from this single allowance and qwen3 cannot be stopped from
# reasoning on this build (see engine/explain.generate). Measured over six real card
# prompts: reasoning ran 775-1838 tokens for an answer of about 90. 320 was the original
# value and every card came back truncated mid-thought.
NUM_PREDICT = 2048

# nomic-embed-text is trained with asymmetric task prefixes. Omitting them puts the
# model off-distribution and measurably corrupts ranking: without these, two driving
# offences scored LESS similar to each other (0.457) than one did to an unrelated pit
# lane speed rule (0.524). With them, the ordering corrects (0.640 vs 0.541).
# See docs/concepts/01-embeddings.md. Never embed raw text.
EMBED_PREFIX_DOC = "search_document: "
EMBED_PREFIX_QUERY = "search_query: "

# --- Retrieval --------------------------------------------------------------
TOP_K = 8            # passages handed to the model

# Reciprocal rank fusion constant. The literature's default is 60, tuned for fusing many
# retrieval runs over large TREC collections, where damping any single system's top ranks
# is the point. Here there are two strong channels over a pre-filtered pool of a few
# hundred passages, and 60 flattens them: rank 1 scores only 1.4x rank 26, so consensus
# mediocrity outranks single-channel certainty. Querying "Article 33.4", BM25 ranked the
# right article 1st and dense ranked it 26th, and the fusion put it 3rd.
#
# Measured over 50 real incidents against the decision they came from:
#
#     K      60     30     20     10      5      2      1
#     MRR   0.621  0.633  0.644  0.681  0.715  0.723  0.731
#
# 10 takes most of the available gain while still damping. NOT a tuned value: that sample
# is the same data Stage 5 evaluates on, so it is a sanity check, not an evaluation.
# Stage 5 settles this on a held-out golden set.
RRF_K = 10

# --- Serving (Stage 4) ------------------------------------------------------
# The model that writes cards in the browser: qwen3's own non-reasoning release, same
# family and size as LLM_MODEL. Measured on the same 8 real card prompts from 4 races,
# streamed, model load excluded:
#
#                          first answer word    card done     tokens   uncited sentences
#     qwen3:4b-instruct    4.8s  (4.2-7.4)       8.9s          106      0 of 24
#     qwen3:4b, think      29.2s (22.2-93.3)     33.0s (-101)  716      3 of 25
#
# Reading all sixteen answers against their sources, faithfulness was a wash: both
# speculated on the same prompt, each mis-attributed one citation. So the thinking model
# buys nothing measurable for 6x the latency. Eight prompts is a sample, not an
# evaluation; Stage 5's faithfulness judge is what settles it. LLM_MODEL stays as the
# terminal CLI's model so Stage 3's numbers remain reproducible.
#
# CARD_THINK=None sends no think flag at all: the instruct model has no reasoning phase to
# route, and Stage 3 showed what think=False does to a model that does.
CARD_MODEL = "qwen3:4b-instruct"
CARD_THINK: bool | None = None

# Ollama unloads a model after 5 idle minutes, and reloading measured 3.6-5.4s. A race has
# quiet stretches far longer than that, so without this the first card after each one
# pays a reload on top of everything else. The server also loads the model at startup.
OLLAMA_KEEP_ALIVE = "30m"

# Race seconds after which an incident is too old to be worth the model's time. See
# server/scheduler.py and the simulation behind it.
PROSE_MAX_AGE = 120.0
