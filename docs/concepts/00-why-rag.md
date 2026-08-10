# 00. Why RAG at all

## The problem

A language model knows what was in its training data, frozen at a cutoff. It does not
know the FIA's 2026 regulations if they were published later, it cannot cite a specific
stewards' decision, and when it does not know something it produces fluent text anyway.

That last property is the dangerous one: the model has no internal signal separating
"I recall this" from "this sounds right".

## Three fixes, and why we chose the third

**Fine-tuning** adjusts the model's weights on your data. It teaches *behaviour and form*
well and *facts* badly. You cannot cite a source, you cannot delete a fact, and every
regulation update means retraining. When the 2027 rulebook lands you start over.

**Long context** stuffs everything into the prompt. Simple, increasingly viable, but our
corpus is ~600 decision documents plus three rulebooks. Far past any practical window,
you pay for every token on every query, latency scales with the stuffing, and accuracy
degrades in the middle of very long contexts. We also have a ~2 second budget.

**Retrieval-augmented generation** keeps knowledge outside the model in a searchable
store, fetches only the passages that matter, and puts those in the prompt. The model
stops being a database and does what it is actually good at: reading and explaining.

## The framing that matters

RAG **separates knowledge from reasoning**.

- Knowledge lives in an index you can inspect, update and delete from.
- Reasoning lives in the model.

That separation buys three things fine-tuning cannot:

| Property | Why it follows |
|---|---|
| **Citations** | You know exactly which document produced the answer |
| **Freshness** | Updating a fact is an index write, not a training run |
| **Auditability** | You can tell whether the wrong doc was retrieved, or the right doc misread |

The third is the one people underrate, and it is the whole basis of our Stage 5 evaluation.

## Honest limitation

RAG *reduces* hallucination, it does not eliminate it. The model can still misread a
correctly retrieved passage, or blend two documents into a claim neither makes.
Retrieval bounds the problem; it does not solve it.

This is exactly why we measure retrieval accuracy and answer accuracy **separately**.

## Interview drill

**Q: When would you fine-tune instead of using RAG?**
When you need a behaviour or output format the base model lacks: a house style, a rigid
schema, a specialised classification. Not when you need it to know facts. They also
compose: fine-tune the form, retrieve the facts.

**Q: Context windows are getting huge. Is RAG obsolete?**
No. Cost scales with tokens processed per query; attention quality degrades over very
long contexts; and stuffing gives you no citation trail. Retrieval is also simply cheaper
for corpora that do not fit regardless.

**Q: What is the failure mode RAG introduces that a plain LLM does not have?**
Retrieval failure. If the right document is never fetched, a perfectly capable model will
give a confidently wrong answer grounded in the wrong source. Your system is now only as
good as its search, which is why retrieval gets its own metrics.
