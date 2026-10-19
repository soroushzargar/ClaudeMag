# Hybrid Reasoning Agents

**arXiv:** 2610.18861  
**Venue:** arXiv preprint, October 2026  
**Authors:** Jiaru Zou, Yunzhe Qi, Sirui Chen, Federico Bianchi, Shang Zhu, Pan Lu, Jingrui He, James Zou  
**Affiliations:** Together AI, Stanford University  
**Date:** October 13, 2026

---

## One-Line Summary

HybridAgent switches between latent-space reasoning and text-based tool interaction within a single generation pass, delivering a 10.4% accuracy gain over text-only and latent-only baselines while using 35–55% fewer tokens.

---

## Problem

Two paradigms exist for building reasoning agents: text-based interaction and latent reasoning. Text-based interaction (standard chain-of-thought with tool calls) is interpretable and natively supports tool use, but each intermediate reasoning token incurs attention and generation cost. Latent reasoning (processing in hidden-state space without decoding) is more compute-efficient, but it cannot call external tools and its reasoning is opaque to users and auditors. Neither paradigm on its own simultaneously achieves efficiency, tool access, and interpretability.

---

## Key Observation: Reasoning and Interaction Require Different Modes

Internal reasoning—planning, hypothesis formation, chain-of-thought—does not need to be decoded into text at every step. Only when the model needs to interact with the environment (call a tool, read a document, present an answer) does it need to produce text. A hybrid that switches modes at the right moments can exploit the efficiency of latent computation while preserving the expressiveness of text interaction.

---

## Why Existing Approaches Fall Short

- **Text-based agents:** All intermediate reasoning is decoded, producing interpretable but expensive token streams; tool calls are native but context length grows with reasoning depth.
- **Latent-only models:** Compute internally without materializing tokens; efficient but cannot call APIs, databases, or search tools; reasoning is inaccessible.
- **Shallow fusion approaches:** Interleave text and embedding layers without a principled switch mechanism; suffer from mode confusion at switch points, losing both the coherence of latent reasoning and the expressiveness of text.

---

## Core Method: HybridAgent

**HybridLink module:** A lightweight adapter, inserted at designated layers of the base transformer, that handles the transition between text mode and latent mode. In text-to-latent direction, HybridLink compresses the recent text context into an initial latent state vector. In latent-to-text direction, it projects the latent state back into the token embedding space for tool call or answer generation.

**Continuous working memory:** A fixed-size buffer of latent vectors that persists across turns. Unlike KV-cache compression, which loses information, the working memory preserves the full hybrid context at constant memory cost; eviction is handled by a recency-weighted update rule.

**Post-training recipe:**
- **HybridSFT:** Supervised fine-tuning on trajectories where ground-truth switch points are labeled (based on when the agent transitions between reasoning and action in the training data).
- **HybridRL:** Reinforcement fine-tuning with a reward that jointly penalizes incorrect answers, excessive text decoding, and tool-call failures, encouraging the model to internalize reasoning and externalize only necessary communication.

---

## Technical Formulation

Let the transformer have $L$ layers and hidden dimension $d$. At each turn $t$, the model maintains:
- A text prefix $x_t \in \mathcal{V}^*$ (visible tokens for tools and user).
- A latent state $z_t \in \mathbb{R}^{d_z}$ (internal reasoning buffer).

The HybridLink at layer $l$ computes the switch:

$$z_t = \text{HybridLink}^{(\text{t}\to\text{l})}(h_l(x_t), z_{t-1}) \quad \text{(text-to-latent)}$$
$$x_t' = \text{HybridLink}^{(\text{l}\to\text{t})}(z_t) \quad \text{(latent-to-text, at tool-call points)}$$

where $h_l(x_t)$ is the hidden state at layer $l$ for the text prefix. The working memory update is:

$$z_{t} = (1 - \alpha) z_{t-1} + \alpha \cdot f(z_{t-1}, h_L(x_t))$$

with $\alpha$ a learned gate value and $f$ a two-layer MLP.

The HybridRL objective is:

$$\mathcal{J}(\theta) = \mathbb{E}\!\left[R_{\text{task}} - \lambda_1 \|x_t^{\text{reason}}\|_1 - \lambda_2 \cdot \mathbf{1}[\text{tool fail}]\right]$$

where $\|x_t^{\text{reason}}\|_1$ is the number of reasoning tokens decoded, and $\lambda_1, \lambda_2$ are regularization weights.

---

## Learning or Inference Procedure

1. **Label switch points** in existing reasoning trajectories (text-to-latent at the start of a reasoning block; latent-to-text before each tool call or final answer).
2. **HybridSFT** trains the HybridLink modules on labeled trajectories.
3. **HybridRL** fine-tunes the full model with the joint reward, further sharpening the switch policy.
4. At inference, the model operates autonomously: when reasoning continues, it stays in latent mode; when a tool call or answer is needed, it switches to text mode.

---

## What the Paper Claims

- **Accuracy:** +10.4% average over existing text-based and latent-only agents across 11 benchmarks.
- **Token efficiency:** 35.5–54.7% fewer total tokens generated compared to text-based baselines at matched accuracy.
- **Cost reduction:** 2.7–3.7× lower inference cost than text-based baselines.
- **Scale robustness:** Gains hold across Qwen3-Base at 4B, 8B, and 14B; the relative advantage increases slightly with model size.

---

## Experimental Findings

- **Models:** Qwen3-Base at 4B, 8B, and 14B parameter scales.
- **Benchmarks (11 total):** Mathematics, science, and medical reasoning domains; tool-use tasks requiring search and API calls.
- **Accuracy vs. text agent:** +10.4% average; highest gains on multi-step tool-use tasks.
- **Accuracy vs. latent-only:** Larger gain; latent-only models cannot complete tool-dependent tasks.
- **Token count:** The hybrid policy produces 35.5–54.7% fewer tokens than text-based chains of equal or lower accuracy.
- **Cost:** 2.7–3.7× cost reduction based on token-count × per-token price.

---

## Ablations and Interpretation

- **Without HybridSFT, only HybridRL:** Convergence is slower and final accuracy is lower, showing that supervised switch-point labeling provides a useful initialization.
- **Without working memory:** Accuracy drops on tasks requiring multi-turn context retention; the model forgets earlier tool outputs stored in the latent buffer.
- **Without latent reasoning:** Degrades to text-only agent; token count rises and accuracy falls below HybridAgent.
- **HybridLink placement:** Inserting at middle layers ($l \approx L/2$) outperforms both early and late placement; early placement loses text context before reasoning begins, late placement provides insufficient latent depth.
- **Why it works:** The model learns that tool-call latency and token cost dominate total inference time; switching to latent mode for the reasoning portion that does not need to be observed is the free-efficiency move.

---

## Reference

Jiaru Zou, Yunzhe Qi, Sirui Chen, Federico Bianchi, Shang Zhu, Pan Lu, Jingrui He, James Zou. **Hybrid Reasoning Agents**. arXiv:2610.18861. October 2026.  
https://arxiv.org/abs/2610.18861
