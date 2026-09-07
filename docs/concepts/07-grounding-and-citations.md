# 07. Grounding, citation binding, and where the time goes

Retrieval selected the passages. This is everything after that: what the model is allowed
to see, what happens to what it writes, and the two things that went wrong in a way worth
recording.

## Grounding is a prompt property, not a model property

The model is given the retrieved passages and told, in terms, that they are the only
information it may use. It never sees the corpus. It is never asked a question whose answer
was not handed to it.

This is worth stating precisely because "RAG prevents hallucination" is the most common
overclaim in the field, and it is false. Retrieval **reduces** hallucination by making the
correct answer available in context, which is a real and large effect. It does not
**eliminate** it, and there are three distinct ways the output can still be wrong:

1. **Retrieval missed.** The right passage was never fetched, so the model answers from
   parametric memory or invents something. Our `--check` says this happens on roughly 36%
   of decisions at k=5, mostly because the International Sporting Code is not in the corpus.
2. **Retrieval succeeded and the model ignored it.** The passage was in the prompt and the
   answer contradicts it. This is the failure people mean by "hallucination despite RAG",
   and it gets worse as the context grows.
3. **The model cites something it was not given.** It writes `[7]` when six sources were
   supplied.

Only the third has a mechanical fix, and we take it.

## Citation binding

The model writes `[2]`. We do not print `[2]` and hope. `bind_citations` resolves every
marker against the sources actually supplied and does three things:

- **Markers that resolve** are kept, and their source is published in the card's footer.
- **Markers that point nowhere** - `[7]` when there were six sources - are **deleted from
  the text**, and recorded in `dropped_citations`.
- **Surviving markers are renumbered** in order of first use, so a card's footnotes read
  1, 2, 3 rather than exposing retrieval's internal ordering.

Deleting rather than rendering is the deliberate part. An invented citation is *worse than
no citation*, because it launders a hallucination into something that looks checkable. A
reader who sees `[7]` assumes a source exists and does not chase it. Removing the marker
leaves an unsupported sentence, which is honest, and `dropped_citations` makes the event
visible to us instead of to the reader.

Note what this does **not** do: it verifies that a cited source *exists*, never that it
*supports the claim*. The model can cite source [1] for a sentence [1] does not contain.
That is a faithfulness question, it needs a judge rather than a regex, and it belongs to
Stage 5.

## Bug 1: the model published its own reasoning

The first version of this stage produced cards like this:

```
Okay, the user wants me to explain an F1 race control message. Let me check
what each source says. Source [1] is about immobilized cars on the starting
grid... but the incident here isn't about the grid - it's during lap 1 with
track yellow. So probably not directly relevant. Source [2] is about official
```

That is the model's private deliberation, printed as the explanation and cut off mid-thought
before any answer arrived.

The code was asking Ollama for `think: False` and stripping `<think>...</think>` from the
response as a backstop. Both failed, and the interaction between them is the lesson.

Measured on Ollama 0.32.9 with `qwen3:4b`:

| request | reasoning appears in | strippable? |
|---|---|---|
| `think: True` | its own `thinking` field | n/a - `response` is clean |
| `think: False` | **`response`, untagged** | **no - there is no tag left** |
| `/no_think` in prompt | `thinking` field (flag ignored) | n/a |

`think: False` is ignored by this build in the sense that matters: the model reasons
anyway. But it *does* suppress the `<think>` tags. So asking for no thinking produced the
worst possible outcome - reasoning still generated, now indistinguishable from the answer,
and the regex that existed to catch it had nothing to match. **The defence was disabled by
the same flag that created the thing it defended against.**

`/no_think`, qwen3's documented soft switch, is a no-op here: 285 tokens against 275 for a
thinking run of the same prompt, through both `/api/generate` and `/api/chat`.

The fix is to ask for `think: True`, which looks backwards for a system that wants no
reasoning and is exactly right: the model is going to reason regardless, so the only real
question is whether the reasoning arrives in a field we can ignore or inline in the text we
publish. The `<think>` regex stays as a backstop for builds that do emit tags.

**The transferable lesson:** when a provider flag suppresses a *marker* rather than a
*behaviour*, disabling the feature can remove your ability to detect it. Verify what the
flag actually did instead of trusting its name - the check took one HTTP request.

## Bug 2: one token budget, two consumers

`num_predict` was 320, chosen as "three short sentences, generously". But `num_predict`
caps reasoning **and** answer together, and measured over six real card prompts the
reasoning ran **775 to 1,838 tokens** to produce an answer of about 90. Every card was
truncated before the answer began.

Set to 2,048, with a guard: an empty `response` plus `done_reason == "length"` raises
`BudgetExhausted` rather than rendering a blank card that looks like a retrieval failure.
`Explainer.explain` catches it and degrades to a card carrying the retrieved citations and
no prose - the citations are the part that carries the value, and a replay that dies on one
slow generation is worse than a card that is merely terse.

## The latency problem, stated honestly

With the fix, cards are correct. They are also slow:

```
retrieve:    0.4 - 1.5 s
generate:     43 - 114 s
```

Retrieval is not the problem. Generation is, and specifically the ~1,100 reasoning tokens
spent to emit ~90 tokens of card. About 92% of the generated tokens are discarded.

The alternative on hand, `llama3:8b`, is a non-reasoning model: median **12.5s**, a 4.6x
improvement. It also speculates - "This is likely due to..." - against a prompt that
explicitly forbids speculation, which for a system whose whole claim is grounded citation
is the wrong trade. Correctness first; qwen3 stays.

Stage 4's budget is ~2 seconds to first card. Neither model meets it, and this is now a
known, measured Stage 4 problem with three options: a smaller non-reasoning model, streaming
for perceived latency, or the template degradation already built for `BudgetExhausted`.

## Caching: the optimisation that mostly is not one

Cards are cached on `Query.signature` - stage, offence, cars, corner, and the rulebook in
force. Deliberately *not* the message text and *not* the lap: the same offence by the same
driver at the same corner is the same explanation on lap 12 and on lap 40.

The reasoning was that a race issues these in bursts. Measured across **12 races of 2026,
542 card-worthy messages**:

| cache key | distinct | hit rate |
|---|---|---|
| as built: stage + offence + **cars** + turn | 485 | **11%** |
| without car numbers | 295 | 46% |
| offence only | 231 | 57% |

**11%.** The assumption was wrong, and the reason is structural: the signature contains car
numbers, and incidents overwhelmingly involve different cars. Dropping cars from the key
quadruples the hit rate - and is not available, because the card names the car ("Car 23
failed to...") and a cached card would name the wrong one.

The deeper point is that **the expensive part is exactly the car-specific part**. Retrieval
barely depends on the car, and could be cached on the car-free key for a 46% hit rate - but
retrieval costs about a second and generation costs a minute, so that saves nothing worth
having. Caching cannot fix this. The latency has to be fixed where it actually is.

Keeping the measurement is the point. An 11% cache is still worth its twenty lines, but it
was justified in advance by a guess about burstiness that the data does not support, and a
"performance optimisation" nobody measures is indistinguishable from a decoration.

## Interview drill

**Q: Does RAG solve hallucination?**
No. It reduces it substantially by putting the correct answer in context, but there are
three separate failure modes: retrieval misses, the model contradicts a passage it was
given, and the model cites a source that does not exist. Only the third has a mechanical
fix. The other two need evaluation, and the useful diagnostic is attributing failure to
retrieval or generation, because the fixes are completely different.

**Q: What does citation binding buy you?**
It converts a claim of grounding into a checkable one. Every marker is resolved against the
sources actually supplied; unresolvable markers are deleted rather than rendered, because a
citation pointing at nothing is worse than none - it makes a hallucination look verified. It
does not verify that a source supports the claim; that needs a judge.

**Q: You cache generated cards. How well does it work?**
11% hit rate over 542 real messages, against my assumption of much more. The key includes
car numbers because the card names the car, and incidents involve different cars. Removing
cars would give 46% and wrong output. The honest conclusion is that caching cannot solve
this latency, because what is expensive is precisely what varies.

**Q: A flag to disable a model's reasoning does not work. How would you find out?**
Send one request and inspect the raw response body rather than the parsed text. Here
`think: False` left the reasoning in `response` with the tags removed, so it looked like a
normal answer and defeated the regex written to catch it. The general habit is to verify
what a provider flag changed, not what its name suggests.
