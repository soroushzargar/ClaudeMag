# Beyond Language Priors: Diagnosing and Fixing Visual-Origin Hallucinations in Multimodal LLMs

**arXiv:** 2609.00231  
**Submitted:** August 31, 2026  
**Authors:** Peiyang Xu, Xiaopei Zhu, Jun Zhu, Xiaolin Hu  
**Affiliation:** Tsinghua University

---

## Headline Finding

Visual-origin hallucination — where multimodal LLM outputs are unfaithful not because of language prior bias but because of incorrect visual feature extraction and image-text misalignment — is a quantitatively significant and under-addressed cause of object hallucination in MLLMs, diagnosable via cosine similarity and Smooth Grad-CAM entropy, and remediable through targeted visual encoding interventions.

---

## Key Findings (Pyramid Layer 2)

1. **A second root cause beyond language priors.** Existing work blames object hallucination predominantly on language statistical biases. This paper provides quantitative evidence that a substantial fraction of hallucinations originates in the visual pipeline itself, not the language decoder.
2. **Lower image-text cosine similarity predicts hallucination.** Hallucinated outputs exhibit systematically lower cosine similarity between image embeddings and generated text embeddings compared to faithful outputs, across multiple MLLM architectures.
3. **Higher visual entropy signals misalignment.** Smooth Grad-CAM entropy — measuring the diffuseness of image regions attended by the model — is significantly higher for hallucinated samples, indicating that the visual encoder fails to focus on the task-relevant objects.
4. **Targeted visual fixes outperform language-side interventions.** When the root cause is visual-origin, interventions at the visual encoding stage (contrastive visual training and attention focusing) reduce hallucination rates more efficiently than post-hoc decoding-side fixes.

---

## Methodology (Pyramid Layer 3)

### Background: The Language Prior Hypothesis

Object hallucination occurs when an MLLM generates text referring to objects not present in the image. The dominant explanation is the **language prior hypothesis**: the language model component has learned strong co-occurrence statistics (e.g., "picnic basket" often appears with "park"), and generates these objects even when absent from the image.

However, this hypothesis implicitly assumes that the visual encoder correctly extracts image features, placing all blame on the language model component.

### Visual-Origin Hallucination

The authors define **visual-origin hallucination** as hallucination caused by upstream failures in the visual pipeline:
- **Visual feature extraction failure:** The encoder produces embeddings that do not capture task-relevant objects.
- **Image-text misalignment:** The cross-modal alignment between image and text embeddings is insufficient, causing the language decoder to proceed without grounding in the actual image content.

### Diagnostic Metrics

**Metric 1: Image-text cosine similarity**
$$\text{sim}(v, t) = \frac{\mathbf{f}_v \cdot \mathbf{f}_t}{\|\mathbf{f}_v\| \|\mathbf{f}_t\|}$$
where $\mathbf{f}_v$ is the image embedding (from the visual encoder) and $\mathbf{f}_t$ is the text embedding of the generated caption. Lower $\text{sim}(v,t)$ indicates worse visual grounding.

**Metric 2: Smooth Grad-CAM entropy**
$$H_{\text{Grad-CAM}} = -\sum_{r} p_r \log p_r$$
where $p_r$ is the normalized Smooth Grad-CAM activation for region $r$. Higher entropy indicates more diffuse attention — the model fails to focus on task-relevant image regions. Low entropy (focused attention) correlates with faithful generation.

### Causal Decomposition

To separate language-prior from visual-origin hallucination, the authors construct a controlled evaluation:

1. For each image-question pair, generate outputs from the MLLM and classify each hallucinated object.
2. Compute $\text{sim}(v, t)$ and $H_{\text{Grad-CAM}}$ for each pair.
3. Use a threshold to classify hallucinations as visual-origin (low $\text{sim}$ or high $H$) vs. language-prior (high $\text{sim}$, low $H$, but object absent from image).

The authors find that 38–52% of object hallucinations across three MLLM architectures are classified as visual-origin, challenging the language-prior-centric framing.

### Targeted Remediation

**Visual Contrastive Training:** For visual-origin hallucinations, the paper proposes training the visual encoder with a contrastive objective that pushes image representations toward the text representations of ground-truth captions and away from representations of negative captions.
$$\mathcal{L}_{\text{contrast}} = -\log \frac{\exp(\text{sim}(v, t^+)/\tau)}{\exp(\text{sim}(v, t^+)/\tau) + \sum_j \exp(\text{sim}(v, t^-_j)/\tau)}$$

**Attention Focusing Regularization:** A regularization term is added to reduce visual entropy during training:
$$\mathcal{L}_{\text{focus}} = H_{\text{Grad-CAM}} = -\sum_r p_r \log p_r$$
This encourages the visual encoder to attend to specific, task-relevant regions.

---

## Technical Formulation

### Hallucination Rate Decomposition

Let $N$ be the total number of hallucinated object instances in an evaluation set. The authors decompose:
$$N = N_{\text{lang}} + N_{\text{vis}} + N_{\text{ambig}}$$
where $N_{\text{lang}}$ = language-prior-origin, $N_{\text{vis}}$ = visual-origin, and $N_{\text{ambig}}$ = ambiguous origin (both factors contribute).

On MSCOCO-Hallu benchmark with LLaVA-1.5:
- $N_{\text{lang}} / N = 41\%$
- $N_{\text{vis}} / N = 44\%$
- $N_{\text{ambig}} / N = 15\%$

### Combined Training Objective

The full training objective for remediating visual-origin hallucinations:
$$\mathcal{L} = \mathcal{L}_{\text{LM}} + \lambda_1 \mathcal{L}_{\text{contrast}} + \lambda_2 \mathcal{L}_{\text{focus}}$$
where $\mathcal{L}_{\text{LM}}$ is the standard language modeling loss, and $\lambda_1, \lambda_2 > 0$ are balancing weights tuned on a validation set.

---

## Experiments

### Benchmarks
- **POPE:** Polling-based Object Probing Evaluation for hallucination rate.
- **MSCOCO-Hallu:** Custom benchmark for fine-grained hallucination categorization.
- **MME:** Multimodal evaluation suite including object-level and attribute-level tasks.
- **LLaVA-Bench:** Open-ended generation quality.

### Results

| Model | POPE (Acc.) | MSCOCO-Hallu (Hallu. Rate) | MME-Perception |
|-------|-------------|---------------------------|----------------|
| LLaVA-1.5 (baseline) | 85.3% | 28.4% | 1,531 |
| + Language-side fix (ICD) | 87.1% | 24.9% | 1,548 |
| + Visual contrastive training | 88.6% | 20.1% | 1,574 |
| **+ Full method (ours)** | **89.4%** | **17.8%** | **1,591** |

The full method reduces hallucination rate by 37.3% relative on MSCOCO-Hallu while improving MME-Perception by 60 points.

---

## Ablations and Interpretation

- **Visual contrastive training alone vs. attention focusing alone:** Contrastive training accounts for 65% of the hallucination rate reduction; attention focusing contributes an additional 35%, and the two are complementary rather than redundant.
- **Applicability to other architectures:** The diagnostic metrics predict visual-origin hallucinations with AUC 0.78–0.83 across three MLLM families (LLaVA, InstructBLIP, MiniGPT-4), suggesting the finding is not architecture-specific.
- **No regression on non-hallucination benchmarks:** The visual contrastive objective does not degrade object recognition accuracy or VQA performance, confirming the fix is targeted.
- **Threshold sensitivity:** The visual-origin classification threshold (on $\text{sim}$ and $H$) affects decomposition proportions but not the qualitative finding that visual-origin accounts for a substantial fraction of hallucinations.

---

## Reference

Peiyang Xu, Xiaopei Zhu, Jun Zhu, and Xiaolin Hu. **Beyond Language Priors: Diagnosing and Fixing Visual-Origin Hallucinations in Multimodal LLM.** arXiv:2609.00231, August 2026. https://arxiv.org/abs/2609.00231
