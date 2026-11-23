# Case Study: Building a Vietnamese Data-Agent LLM Under Hard Production Constraints

*How I fine-tuned a compact model stack to read and answer questions over messy real-world
business data — fully offline, deterministic, and under a strict parameter budget.*

> **TL;DR** — I built an end-to-end agent that answers natural-language **Vietnamese** questions
> over real small-business data (Excel / CSV / SQLite). It had to fit in **≤15B parameters**, run
> **fully offline** on **2 GPUs**, and stay **reproducible within 0.03 score drift** between runs.
> The result: a robust, production-grade system — not a notebook demo. This is the level of rigor
> I bring to every fine-tuning engagement.

---

## The challenge

A national Vietnamese AI competition posed a deceptively hard problem: given an arbitrary small
shop's data files (spreadsheets, CSVs, SQLite databases) and a Vietnamese question, return the
correct answer — or **abstain** when unsure (wrong answers were penalized).

The constraints are what made it an *engineering* problem, not a prompting exercise:

- **≤15B total parameters** across every model, adapter, router and classifier.
- **Fully offline** on the evaluation machine (2 GPUs) — no internet, no external APIs.
- **Reproducible**: two runs on the same input had to differ by ≤0.03 in score.
- **No training on the test data** — generalization had to be real.
- **Messy inputs**: inconsistent headers, mixed encodings, typos, diacritic-stripped text,
  and "teencode" Vietnamese.

## The architecture

Rather than throw one giant model at it, I split the work into a disciplined pipeline:

```
raw files → compression + column/row-role hints (classical ML, no embeddings)
          → 9B model drafts a structured spec of the data  → validate & pick best
          → execute spec → canonical tables → data card
          → per question: 9B emits intent (JSON) + 4B emits code  → execute both
          → confidence voting + thresholds → answer or abstain
```

Two fine-tuned models carried the load: a **9B** model for schema/intent generation and a
**4B** model for code synthesis — together under the 15B budget.

## Engineering highlights

- **A 3-gate label-quality pipeline.** Every training label (spec, intent, code) had to pass:
  (1) valid schema + runs within time limit, (2) matches the gold answer under the official
  normalization rules, (3) **still correct on 5 corrupted variants of the data**. Thousands of
  labels, each triple-checked — this is why the model generalized instead of memorizing.
- **A compact fine-tuned stack.** LoRA on a 9B + 4B model pair, with the image/MTP modules
  excluded from adaptation, merged to hit an exact parameter budget.
- **Robustness by design.** The system handled stripped diacritics, Telex typos, teencode, odd
  status strings, footer/total rows, and multi-file sources — validated with automated fuzzing.
- **Determinism under serving.** Getting two runs to agree within 0.03 on non-batch-invariant
  kernels required serializing generation at the spec stage and controlling sampling — the
  difference between a research result and something you can actually ship.
- **Calibrated abstention.** Confidence = agreement across independent votes (model intent vs.
  independently generated code), with thresholds that rise when the data looks unreliable — so
  the system stays quiet instead of guessing wrong.

## Results

A deterministic, fully-offline, sub-15B solution that reads and answers over **arbitrary**
small-business datasets it had never seen — with measured, reproducible quality and principled
abstention. Every design decision was validated by held-out evaluation, not vibes.

## From a competition to your business

The same pipeline is exactly what a company needs when a generic model (GPT/Claude) underperforms
on *their* domain:

- You have proprietary data and want a model that speaks your format, tone, or schema.
- You need it to run **on-prem / offline** for privacy or cost.
- You need **measured** quality and safe behavior, not a hopeful prompt.

I turn that into a working, evaluated model — data prep → fine-tuning → eval → deployment —
and I own the hardware (2×H100), so your data never leaves a machine I control.

**Proof it's a real pipeline, not slides:** a runnable mini-version of this flow (load → measure
before → LoRA fine-tune → measure after → deliver adapter + report) is available as a live demo —
it took a 4B model from **0% to 100%** exact-match on a Vietnamese structured-output task in
**11 seconds** on one H100.

## Tech stack

PyTorch · Transformers · TRL (SFT, preference tuning) · PEFT (LoRA/QLoRA) · DeepSpeed ZeRO ·
vLLM · classical ML (TF-IDF/LinearSVC/LightGBM) for feature hints · NVIDIA H100.

---

### Work with me

I fine-tune open LLMs (Llama, Qwen, Mistral, Gemma) on your data and deliver production-ready,
evaluated models. Specialties: **Vietnamese & multilingual NLP, structured output, robustness,
on-prem deployment.**

- 📫 [your email / LinkedIn / Upwork / Fiverr link]
- 🔗 Live demo pipeline: https://github.com/leganvy/llm-finetune-demo

*(Note: a Vietnamese version of this article is available for Viblo / local community.)*
