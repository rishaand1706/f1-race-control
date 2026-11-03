# 09. Agents, tools, and the guardrails that make one safe to ship

Stages 3 and 4 are a **pipeline**: the code decides what to retrieve, three filtered
searches, and the model only writes prose over what it is handed. This stage rebuilds the
same card as an **agent**: the model is given search tools and decides for itself whether to
search, what to search for, and when it has enough. The question is not whether that is more
fashionable. It is what the model's new freedom costs, measured, and which of those costs
code can take back.

## Pipeline vs agent, precisely

| | pipeline (Stage 3/4) | agent (this stage) |
|---|---|---|
| who decides what to retrieve | code: three slots, always | the model, per incident |
| who writes the query | `engine/query.py`, from facets + race state | the model, in its own words |
| who sets the filters | code: season, rulebook issue, published-before | depends: that is the design question |
| model calls per card | 1 | 2 or more: one per tool round, plus the answer |

An agent is a loop: the model replies either with text or with a request to call a tool;
the runtime calls the tool, appends the result, and asks again. Everything else in this
stage is about what to let the model control inside that loop.

## Why Semantic Kernel

It is Microsoft's agent SDK, and it separates the parts cleanly enough to reason about:

- **Kernel**: the container for services (the chat model) and plugins (the tools).
- **Plugins**: a plain Python class whose methods carry `@kernel_function`. Their type hints
  and descriptions become the JSON schema the model sees. The schema is the contract: a
  parameter that exists in the schema is a parameter the model will fill in.
- **Function choice behavior**: `Auto()` lets the model decide whether to call a tool.
  SK runs the call loop itself.
- **Filters**: middleware around every function call and every automatic tool round.
  This is where guardrails live that must hold whatever the model writes.
- **Agents**: `ChatCompletionAgent` wraps the loop with instructions and a thread.
- **Telemetry**: SK emits OpenTelemetry spans following the GenAI semantic conventions
  (`invoke_agent`, `chat <model>`, `execute_tool <function>`) once
  `SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS=true` is set.

The model stays local (`qwen3:4b-instruct` in Ollama), so the agent costs nothing to run and
every number here can be reproduced offline. Swapping the chat service for Azure OpenAI is a
one-line change to the kernel; nothing else in the agent depends on which model answers.

## The three questions

1. **Does the model search at all, or answer from memory?** A pipeline cannot skip
   retrieval. An agent can, and a 4B model has read about F1.
2. **When the model writes its own query, does retrieval get better or worse?** The
   pipeline's query is built from structured facets. The model's is a paraphrase.
   Measured on the same 120 decisions as `engine.retrieve --check`, same metric: did the
   article the stewards cited come back in the top 5 of a rule search.
3. **What does the loop cost?** At least two model calls instead of one. OpenTelemetry
   spans split each card's time into model time and tool time.

## The naive agent, and what should go wrong

Built the way a tutorial builds one: a short instruction to search first, and tools whose
parameters include the filters, `search_rules(query, season, date)`, because the
regulations really are versioned by date and it seems natural to let the model say which.
Precedent search has no published-before filter, because nothing in the tool's schema
asks for one.

Predictions, written before running it:

- Some answers will skip retrieval and quote an article from memory, unchecked.
- Some rule searches will use a wrong season or date, re-creating Stage 3's wrong-rulebook
  bug from the model's side.
- Precedent search will sometimes return the stewards' ruling on the very incident being
  explained: Stage 3's leak, re-introduced because the filter is no longer code.
- The model's paraphrased queries will score below the pipeline's 64.2% hit@5, because they
  drop the identifiers BM25 depends on.

## The guarded agent

Four rules, each enforced by code rather than by asking the model nicely:

1. **The model writes the query; the code sets the filters.** The tools' schemas have no
   season, date or published-before parameter. They close over the incident's own, so
   the model cannot choose a wrong rulebook because it is never asked to choose one.
2. **No retrieval, no answer.** If a run ends without a rule search, the agent is told so
   and gets one retry. If it still has not searched, the card is not published as prose.
3. **Cite only what this run retrieved.** Every passage any tool returns goes into a ledger
   and keeps one number for the whole run. Citation binding (brief 07) resolves markers
   against the ledger, so a marker for a passage the tools never returned is deleted.
4. **A bounded loop.** An auto-function-invocation filter stops the loop after a fixed
   number of tool calls, so a model that keeps searching cannot run up latency without limit.

Rule 3 checks that a cited source exists, not that it supports the claim: the brief 07
caveat still applies, and Stage 5's judge is still the answer to it. What this stage adds is
a check on article numbers written in prose: an "Article 33.1" that appears in no retrieved
passage is counted as unsupported, whatever marker sits next to it.

## What happened

Measured with `python -m eval.agent_check` on the same 120 decisions as
`engine.retrieve --check`, both variants over all 120.

| | naive | guarded |
|---|---|---|
| answered after a rule search | 0% | 100% |
| ...only because the retry guard fired | - | 39.2% |
| rule hit@5, the agent's own search | 0% | 70.0% |
| rule hit@5, the pipeline on the same rows | 64.2% | 64.2% |
| rule searches against the wrong rulebook | - | 0% |
| cards with a citation marker pointing at nothing | 100% | 0% |
| cards naming an article never retrieved | 63.3% | 0% |
| tool-call cap reached | 0% | 0% |
| card time, median | 5.3 s | 11.3 s |
| model calls per card, mean | 1.00 | 2.40 |

Against the predictions:

- **Skipping retrieval was not "some answers".** Told only to search first, the naive agent
  never searched, on any of the 120. It answered from what a 4B model remembers about F1:
  every card carried a citation marker with nothing behind it, and 76 of the 120 named an
  article it had never been shown.
- **The wrong-rulebook and leak predictions could not be tested on the naive agent**,
  because it never called a tool. In the guarded agent they are ruled out by construction
  (rule 1), and the counters agree: no wrong-rulebook search, no leaked ruling.
- **The paraphrase prediction was wrong.** The agent's own queries found the cited article
  in the top 5 for 70.0%, against 64.2% for the pipeline. That is seven decisions, on rows
  that are not held out, so it is not a win; what it does rule out is the large loss the
  prediction expected.
- **The retry guard earns its place**: 47 of the 120 cards searched only because it fired.

The cost is the loop: 2.40 model calls per card and an 11.3 s median, 96% of it inside the
model according to the OpenTelemetry spans. And one gap the guards do not close: the agent
almost never asks for precedents (1 of 120 cards), so the slot the pipeline always fills is
usually empty. That is a prompt and tool-design problem, not a safety one, and it is open.
