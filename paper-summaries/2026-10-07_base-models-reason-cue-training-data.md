# Base Models Can Reason By Taking a Cue From Training Data

**arXiv:** 2610.06851  
**Authors:** Sophie L. Wang, Amil Dravid, Rulin Shao, Kevin Farhat, Sewon Min, Alexei A. Efros  
**Affiliation:** UC Berkeley  
**Date:** October 5, 2026

---

## One-Line Summary

Simple starting-token cues in a base language model's response—derived causally from training data—can raise math reasoning accuracy by over 35 percentage points, matching RL-trained counterparts without any fine-tuning.

---

## Problem

Large language model post-training pipelines rely on reinforcement learning with verifiable rewards (RLVR) to elicit extended reasoning chains that dramatically improve performance on math and coding tasks. A natural question is: what exactly does RL teach, and can it be replicated more cheaply? This paper investigates whether the improvements attributed to RL training can instead be achieved by manipulating the training-data associations that base models already carry—specifically, by choosing the starting tokens of a model's response.

---

## Why Existing Approaches Fall Short

- **RL post-training is expensive:** RLVR requires generating many rollouts, computing verifiable rewards, and running multiple gradient updates, making it costly relative to the size of performance gains sought.
- **Prompt engineering is empirical:** Prior work has shown that chain-of-thought prompts can boost performance, but the mechanism is poorly understood. It is unclear whether the effect is semantic (the prompt conveys useful instructions) or statistical (the prompt selects a region of the training data distribution with favorable behavior).
- **Base model capabilities are underexplored:** The default response-start tokens a practitioner chooses are often arbitrary, even though they condition the entire continuation via the model's next-token predictions.

---

## Core Method

The paper introduces the concept of a **reasoning cue**: a short prefix (often a single token or punctuation sequence) placed at the very start of a base model's response. The key claim is that these cues act not as semantic instructions but as distributional selectors—they direct the model toward regions of its training data where careful step-by-step reasoning was associated with the same tokens.

**Evaluation setup:** For a given cue, the model's MATH-500 pass@1 and code-generation accuracy are measured, sweeping over a vocabulary of candidate cues.

**Causal intervention protocol:** To confirm that the effect of a cue is mediated by training data:
1. Identify documents in the training corpus that are preceded by the cue token.
2. Remove or add those documents from a fine-tuning mix.
3. Measure whether the cue's reasoning effect disappears or appears correspondingly.

This intervention confirms a causal link: the cue's efficacy is causally traceable to the associated training data, not to any semantic meaning of the token itself.

---

## Technical Formulation

Let $\mathcal{M}$ be a base LLM with vocabulary $\mathcal{V}$. For a problem $q$ and a starting token $c \in \mathcal{V}^*$ (a short cue string), define the cue-conditioned response:
$$r_c(q) = \arg\max_{r} \log p_\mathcal{M}(r \mid q, c)$$

The paper measures the expected performance metric $\mathbb{E}_q[\text{Correct}(r_c(q))]$ as a function of $c$.

For the causal intervention, let $D_c \subseteq \mathcal{D}_\text{train}$ be the subset of training documents where the response begins with $c$. Fine-tuning on $\mathcal{D}_\text{train} \setminus D_c$ removes the cue's effect, while fine-tuning on $\mathcal{D}_\text{train} \cup \widetilde{D}_{c'}$ (where $\widetilde{D}_{c'}$ associates an arbitrary token $c'$ with step-by-step reasoning) creates a new effective cue.

---

## Learning or Inference Procedure

**Finding effective cues:**
1. Enumerate a large set of candidate starting tokens/phrases.
2. For each cue, generate responses from the base model on a held-out problem set.
3. Score responses and rank cues by their mean accuracy.

**Causal confirmation:**
1. Identify training documents associated with effective cues.
2. Remove those documents and retrain (or fine-tune) a small model.
3. Verify that removing associated training data degrades the cue's effect.

**Creating custom cues:**
1. Collect step-by-step reasoning traces.
2. Prepend an arbitrary token to each trace.
3. Fine-tune the model on these modified traces.
4. The arbitrary token becomes an effective reasoning cue.

---

## What the Guarantee Says

The paper makes an empirical rather than formal guarantee. The central claim is: for all tested base models and diverse starting-token cues, the reasoning benefit of a cue is causally traceable to its associated training data. The causal intervention—modifying the training data to add or remove reasoning traces behind a cue—produces the corresponding addition or removal of the cue's effect on downstream performance.

---

## Experimental Findings

- **".\n\nOkay"** raises Olmo-3-7B's MATH-500 pass@1 from **42% to 78%**, matching the RL-trained counterpart.
- **"Alright,"** raises Qwen3-14B's MATH-500 from **72% to 87%**, also competitive with RL.
- The same effect transfers to code generation, with analogous cues improving pass@k metrics.
- Causal interventions confirm that adding or removing the relevant training documents causally modulates the cue's effect.
- An arbitrary word ("chicken") becomes an effective reasoning cue after fine-tuning on reasoning traces prefixed with it.

---

## Ablations and Interpretation

**Semantic vs.\ distributional hypothesis:** Replacing the effective cue with semantically similar phrases that do not appear in the relevant training data produces no reasoning gain, ruling out the semantic hypothesis.

**Context length:** The cue effect is stable even when problems are presented with few-shot examples, suggesting the cue operates as a distributional anchor rather than via attention to immediate context.

**Multiple cues:** Different cues elicit different reasoning styles; combining multiple effective cues does not additively compound gains, suggesting they tap into overlapping regions of the training distribution.

**Implications for RL:** Since base-model cue performance is competitive with RLVR, the paper argues that a significant portion of what RLVR learns is to reliably start responses in the region of the distribution that already contains step-by-step reasoning—not to teach the model new reasoning skills.

---

## Reference

Sophie L. Wang, Amil Dravid, Rulin Shao, Kevin Farhat, Sewon Min, Alexei A. Efros.
**Base Models Can Reason By Taking a Cue From Training Data.**
arXiv:2610.06851, October 2026.
https://arxiv.org/abs/2610.06851
