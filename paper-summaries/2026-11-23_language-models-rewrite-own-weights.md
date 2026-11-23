# Language Models Can Rewrite Their Own Weights

**Authors:** Jiajun Fan, Neil He, et al.
**arXiv:** 2611.21885 (cs.LG, cs.AI)
**Submitted:** 16 November 2026
**Venue:** Preprint

---

## Headline

A language model can read its own weight matrices, diagnose failure modes, and edit those weights directly — without gradient descent — producing repairs that generalize to benchmarks the model never trained on.

---

## Key Findings

- A 27B language model, given access to its own weight matrices as text, can identify which matrix is responsible for a failure and propose corrections that restore normal behavior.
- Self-rewriting transfers across modalities: repairs learned from text math problems improve performance on six image benchmarks the model never saw, suggesting the model computes a general diagnostic rather than task-specific memorization.
- The process runs on a single GPU in a few hours.
- Self-rewriting capability scales with model size: larger models diagnose failures more accurately and produce more effective weight corrections.
- The approach requires no gradient computation, no labeled repair dataset, and no access to the original training procedure.

---

## Motivation and Problem

Large language models are fixed after training. If a model's weights encode a failure — a corrupted matrix, a systematic bias, a miscalibrated attention pattern — the only standard remedies are fine-tuning (which requires labeled data and compute) or full retraining. Both are expensive and may introduce regressions elsewhere.

This paper asks a more provocative question: can the model repair itself by reading and editing its own weights, the way a programmer debugs and patches code? Language models already manipulate complex structured text; weight matrices are structured numerical arrays. If a model understands what weights *do*, it can in principle reason about what they *should* contain.

---

## Method: Self-Rewriting

### Weight Matrix Representation

The model is given read access to its own weight matrices in a serialized numerical format. It is prompted to:
1. Describe what the matrix computes (its functional role in the network).
2. Identify anomalies — values that deviate from expected statistical properties or produce identifiable error patterns.
3. Propose corrected values or correction operations.

The corrections are applied directly by overwriting the targeted matrix elements, without backpropagation.

### Diagnostic Protocol

A key step is *failure attribution*: given a set of model behaviors that are wrong, which weight matrices are responsible? The model uses a chain-of-thought diagnostic that:
- Traces a failing input through the layer stack
- Identifies layers where the computation diverges from expected behavior
- Narrows to individual matrices within those layers

This is analogous to symbolic program debugging: the model reasons about data flow through its own computation graph.

### Self-Rewriting Update

Let $W_l$ denote the weight matrix at layer $l$ identified by the diagnostic. The model produces a correction $\Delta W_l$ as a structured text description (e.g., "multiply rows 42–47 by 0.3") which is parsed and applied:

$$W_l \leftarrow W_l + \Delta W_l$$

No gradient or optimizer state is updated. The correction is local to the identified matrix; all other weights are unchanged.

---

## Technical Formulation

### Failure Attribution as a Reasoning Problem

Given a task $T$ where the model fails (output $\hat{y} \neq y^*$), the model constructs an attribution chain:

$$\text{Failure}(\hat{y}) \rightarrow \text{Layer}(l^*) \rightarrow \text{Matrix}(W_{l^*}) \rightarrow \Delta W_{l^*}$$

This is a multi-step inference problem: each arrow corresponds to a reasoning step in the model's chain of thought.

### Generalization via Structural Reasoning

The key theoretical claim is that weight-correction reasoning is *structurally compositional*: the same correction pattern (e.g., "attenuate the dominant singular directions in the value matrix") applies to functionally similar matrices across different input distributions. This explains the cross-modal generalization: the correction learned for text math matrices applies to visually analogous attention structures in the image processing layers.

Formally, if $f: \mathcal{W} \rightarrow \mathcal{C}$ maps weight states to corrections, the claim is that $f$ depends on structural properties of $W$ (singular value distribution, sparsity pattern, norm) rather than on the specific task distribution used to identify the failure.

---

## Experimental Findings

### Weight Corruption Recovery

The authors inject synthetic corruptions into specific weight matrices (scaling a matrix by a large constant, zeroing out rows, swapping columns). The model is asked to recover the matrix without being told what the corruption was.

| Corruption type | Recovery rate (27B) |
|---|---|
| Row zeroing | 94% |
| Column swap | 88% |
| Magnitude scale | 91% |
| Random noise injection | 73% |

Recovery rate is defined as restoring task performance to within 5% of the uncorrupted baseline.

### Cross-Modal Transfer

After learning to repair text-math failures, the model repairs visually analogous structures in image-processing layers. Performance on six held-out image benchmarks improves by 2.1–4.8 percentage points on average after the text-math repair protocol, with no additional image examples.

### Scaling

Self-rewriting diagnostic accuracy (correctly attributing failures to the right matrix) scales approximately log-linearly with model size from 7B to 70B, confirming that larger models are better self-diagnosticians.

---

## Significance

Self-rewriting opens a new paradigm for model maintenance: instead of fine-tuning or retraining, models can be deployed with a self-repair protocol. This is especially relevant for long-lived deployments where failure modes accumulate over time and the original training data or compute is unavailable. The cross-modal transfer result suggests that weight-editing reasoning taps a general structural understanding of neural networks, not just task-specific pattern matching.

The key limitation is that self-rewriting currently addresses failures detectable through the model's own reasoning — systematic bias or corrupted matrices with identifiable statistical signatures. It does not address failures the model cannot introspect, such as unknown unknowns in its knowledge.
