# LLMs Get Lost In Multi-Turn Conversation

### Multi-Turn Performance Collapse Across All Frontier Language Models

**Authors:** Philippe Laban, Hiroaki Hayashi, Yingbo Zhou, Jennifer Neville (Salesforce AI Research)
**Venue:** ICLR 2026 Outstanding Paper Award
**arXiv:** [2505.06120](https://arxiv.org/abs/2505.06120)
**Submitted:** May 9, 2025

---

## Layer 1 — The Big Picture

**What problem does the paper solve, and how does it solve it?**

Large Language Models are predominantly evaluated in the single-turn, fully-specified instruction setting, yet in practice users interact with them across extended multi-turn conversations where initial intent is often underspecified. This mismatch means benchmarks fail to capture a critical failure mode: how LLMs degrade when conversation context grows and user intent drifts or evolves.

This paper performs a large-scale simulation study comparing LLM performance in single-turn versus multi-turn conversation on the same underlying tasks. Across more than 200,000 simulated conversations, the authors find that all evaluated frontier models — both open- and closed-weight — suffer significant performance degradation in multi-turn settings. The average drop across six generation tasks is **39%**, and crucially, once a model takes a wrong turn, it fails to recover — it "gets lost."

---

## Layer 2 — Under the Hood

**Intuition behind the method.**

The authors decompose the observed multi-turn performance drop into two components:

1. **Loss in aptitude**: The model's underlying ability to complete the core task degrades slightly when surrounded by conversation history — smaller effect.
2. **Increase in misalignment losses**: The dominant effect — as conversation progresses, the model increasingly loses track of the user's precise intent, even when the intent has been explicitly stated earlier in the context.

The simulation framework creates "underspecified" conversation scenarios where a user gradually refines a task through dialogue, and the LLM must integrate all turns to arrive at the correct final answer. The experiment isolates turn count as the independent variable, holding all other factors constant.

**Why models get lost**: Early wrong decisions become self-reinforcing. The LLM's own prior outputs form part of the context, and if those outputs contain a misconception about user intent, subsequent turns anchor to that misconception rather than correcting it.

---

## Layer 3 — The Math

**Formal setup.**

Let $T = (t_1, t_2, \ldots, t_n)$ be a conversation of $n$ turns, where $t_1$ is a partially specified initial request and subsequent turns $t_k$ add clarifications. The task label is $y$. Model performance is:

$$\text{Perf}_n = P(f(T_{1:n}) = y)$$

where $f$ is the LLM output. The paper observes:

$$\text{Perf}_n < \text{Perf}_1 \quad \text{for all tested models, all } n > 1$$

The gap grows approximately linearly with conversation length for the first $n \approx 5$ turns, then plateaus. Performance recovery (i.e., $\text{Perf}_{n+k} > \text{Perf}_n$ after a clarification at step $n$) is essentially zero, establishing the "getting lost" property formally.

The paper defines a *recovery rate* $R_n$:
$$R_n = \frac{1}{K} \sum_{k=1}^{K} \mathbf{1}\left[\text{Perf}_{n+k} > \text{Perf}_n\right]$$
and shows $R_n \approx 0$ across all model families and task types.

---

## Experiments & Datasets

**Tasks (6):** Summarization, email drafting, code generation, question answering, creative writing, data extraction.

**Models:** GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro, Llama 3.1 (70B/405B), Mistral Large — all frontier models at time of writing.

**Scale:** 200,000+ simulated conversations with automated scoring.

**Key result:** 39% average drop single-turn → multi-turn. Code generation and QA show the largest drops (~45%); creative writing shows the smallest (~28%). No model tested is robust to multi-turn degradation.
