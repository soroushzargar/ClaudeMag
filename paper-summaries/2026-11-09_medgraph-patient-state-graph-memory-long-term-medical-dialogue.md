# MedGraph: Maintaining Patient State with Graph Memory for Long-Term Medical Dialogue

**arXiv:** 2611.08433  
**Authors:** Donghua Zhang, Zizheng Wang, Ze Dong, Yiding Feng, Siyu Qian  
**Venue:** cs.CL / cs.AI  
**Submitted:** 6 November 2026  
**URL:** https://arxiv.org/abs/2611.08433

---

## Summary

LLM-based medical dialogue assistants fail in longitudinal settings — conversations spanning multiple visits over weeks or months — because their memory systems conflate old and new clinical states, mix facts from different patients, or silently omit safety-critical information. Through a systematic error analysis of five memory baselines on longitudinal medical dialogue, MedGraph identifies three failure modes: outdated state (the assistant reports a resolved symptom as active), cross-patient contamination (facts from an earlier conversation pollute the current patient's context), and missing safety facts (a known allergy or contraindication is not retrieved). MedGraph introduces a three-layer hierarchical graph memory that eliminates all three through explicit state typing, graph-based fact isolation, and mandatory safety-fact attachment.

---

## Method

**Layer 1 — Raw dialogues.** The full transcript of each session is stored verbatim, enabling exact retrieval of clinical reasoning chains when needed.

**Layer 2 — Clinical fact graph.** A typed knowledge graph over entities (symptoms, diagnoses, medications, test results, procedures) with typed relations (has-symptom, prescribed, resolved, contraindicated-with). Each entity carries a temporal validity tag: *active*, *resolved*, or *historical*. When a new mention arrives, a type-specific update rule applies:
- If the entity is new, it is added as *active*.
- If it matches an existing *active* entity with a "resolution" signal (e.g., "symptom resolved", "medication discontinued"), its tag is changed to *resolved* rather than creating a duplicate.
- If the mention is a repeated observation with no state change, the entity is updated in place without changing its tag.

**Layer 3 — Patient profile (safety block).** A compact structured profile is maintained per patient, containing recorded allergies, contraindications, chronic conditions, and emergency contacts. This block is *unconditionally prepended* to every query context — it is never left to retrieval. This prevents safety facts from being silently missed when the retrieval module ranks them below other context.

**Graph isolation.** Each patient has a fully separate graph instance. Cross-patient contamination is impossible by construction.

---

## Experimental Setup

Benchmark: MedMemoryBench — a longitudinal medical dialogue benchmark with multi-visit transcripts, manual ground-truth entity states, and clinical reasoning questions.  
Backbones: GPT-4o-mini, Qwen2.5-72B.  
Baselines: flat vector memory, sliding-window memory, retrieval-augmented memory, Zep (entity memory), MemoryOS.  
Metric: overall clinical accuracy (entity state, diagnosis, medication, safety-fact recall).

---

## Results

| Backbone | Strongest Baseline | MedGraph | Improvement |
|---|---|---|---|
| GPT-4o-mini | — | — | +10.5 pp |
| Qwen2.5-72B | — | — | +10.4 pp |

MedGraph achieves the highest overall clinical accuracy with both backbones, 10.4–10.5 percentage points above the strongest of five baselines.

---

## Key Insight

The central insight is that clinical entities do not merely accumulate — they transition through typed states (active → resolved → historical), and conflating these states is the primary failure mode of flat memory systems. Making transitions explicit through typed graph updates allows the system to maintain an accurate present-tense view of patient state without hallucinating continuity of resolved conditions or missing state changes.

---

## Limitations

- Clinical entity extraction relies on an NLP pipeline; extraction errors propagate directly into the graph and may not be caught by the update rules.
- The safety-block attachment strategy ensures safety facts are present in context, but does not guarantee the LLM attends to them; instruction following on safety blocks is not analyzed.
- MedMemoryBench covers a limited set of clinical specialties; generalization to, e.g., oncology or complex polypharmacy settings is not evaluated.
