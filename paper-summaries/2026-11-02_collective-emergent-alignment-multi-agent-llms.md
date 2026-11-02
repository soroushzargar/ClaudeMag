# Collective Emergent Alignment: Measuring and Amplifying Interaction-Induced Reasoning Alignment in Multi-Agent LLMs

**arXiv:** 2610.34836  
**Authors:** Boxuan Wang, Tianxiang Sun, Linyang Li, Xipeng Qiu, Hang Yan  
**Submitted:** October 26, 2026  
**Venue:** arXiv preprint (cs.AI)  
**URL:** https://arxiv.org/abs/2610.34836

---

## Summary

Multi-agent LLM systems—where several models debate, critique, or collaborate—often converge on better answers than any individual agent achieves alone. But is this collective gain real, or is it just individual self-refinement dressed up as collaboration? This paper draws a sharp distinction and introduces tools to measure and amplify the genuine collective contribution.

The authors define **Collective Emergent Alignment (CEA)** as the improvement in reference-grounded reasoning that arises specifically from inter-agent interaction, over and above what matched individual self-refinement would yield. They introduce **REMAP (Reference-Entailment-Matrix for Agentic Process)**, a measurement framework that tracks which factual claims become grounded in the reference answer at each communication round, attributing each grounding event to either individual reasoning or peer interaction.

---

## Method

**Collective Emergent Alignment (CEA).** At each communication round, REMAP records the set of reference-supported claims in each agent's response. A claim is attributed to CEA if it first appears after an inter-agent communication event and is not recoverable by the agent in self-refinement at matched compute. The CEA score aggregates these events across rounds, normalized by the total information needed for a correct answer.

**REMAP Framework.** REMAP uses a reference-entailment matrix: a binary matrix indexed by (agent, round, fact) that records whether each fact is entailed by each agent's response at each round. The matrix is populated using an entailment model. From this matrix, Semantic Coverage (fraction of reference facts covered by the collective) and per-round alignment gains are computed.

**Gap-Aware Factual Communication.** The paper also proposes a practical communication method: instead of exchanging full reasoning traces (which are long and redundant), agents exchange compact factual messages that highlight specific gaps in each other's reasoning. This is implemented by having each agent identify which of its peer's facts contradict or are absent from its own response, and summarizing only those gaps.

**Experimental Setup.** Seven model families are tested across two multi-hop reasoning benchmarks (MuSiQue and 2WikiMultiHopQA), with group sizes of 2, 4, and 8 agents. Both synchronous (all agents communicate simultaneously) and sequential (agents communicate in a chain) configurations are evaluated.

---

## Results

- CEA is consistently **positive and statistically significant** across all model families and both benchmarks.
- The average CEA score across all settings is **+6.3 F1 points** above matched individual self-refinement.
- **Gap-aware factual communication** amplifies CEA by a further **+3.1 points** on average, relative to full-trace exchange, while **reducing communication tokens by 58%**.
- Larger groups (8 agents) show higher CEA but with diminishing returns; the marginal gain from adding the 8th agent is roughly half that of adding the 2nd.
- Sequential communication yields slightly lower CEA than synchronous but is more robust to one agent dominating the conversation.

---

## Significance

The paper provides the first rigorous decomposition of multi-agent LLM improvement into individual and collective components. This is not just a theoretical exercise: the REMAP framework is a practical diagnostic tool that can be applied to any multi-agent system to diagnose whether inter-agent communication is actually helping. The gap-aware communication method, which cuts tokens by 58% while improving performance, has direct practical implications for the growing ecosystem of multi-agent LLM frameworks. The finding that CEA is real and amplifiable challenges the argument that multi-agent systems are redundant when strong individual models are available.

---

## Limitations

- The REMAP entailment model may make errors on complex factual claims; its precision/recall on the benchmarks is not separately evaluated.
- Gap-aware communication requires the ability to detect contradictions between reasoning traces, which currently uses an auxiliary LLM step and adds latency.
- Benchmarks are limited to multi-hop question answering; CEA in other task families (code, math, planning) is unknown.
