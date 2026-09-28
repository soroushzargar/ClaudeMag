# Hallucination Neurons and Where to Find Them: An Investigation into the Existence of Hallucination Neurons

**arXiv:** 2609.29781  
**Authors:** Not specified in available metadata  
**Date:** September 2026

---

## One-Line Summary

A five-step diagnostic protocol reveals that claimed "hallucination neurons" in Gemma 3 4B are highly correlated with other features, have moderate bootstrap stability, and are not uniquely localized—showing that sparse predictive structure coexists with non-unique neuron selection.

---

## Problem

Mechanistic interpretability increasingly relies on sparse probing methods that identify small sets of neurons claimed to detect and causally influence LLM behaviors such as factuality recall, safety alignment, and hallucination. Claims about "hallucination neurons" (H-Neurons) imply that hallucinatory behavior is localized to identifiable, specific neurons that could serve as targets for auditing and behavioral steering. However, these localization claims are rarely tested against known failure modes of L1-regularized probing in correlated, high-dimensional feature spaces.

---

## Why Existing Approaches Fall Short

- **L1-regularized probing** (Lasso-style neuron selection) is known to select non-unique solutions in the presence of correlated features; the selected neuron set depends on regularization strength and initialization.
- **Reported high probe AUROC** does not imply unique causal localization—many alternative neuron sets may achieve similar or better discrimination.
- **Prior H-Neuron work** (e.g., H-Neurons: On the Existence, Impact, and Origin of Hallucination-Associated Neurons) demonstrates predictive neurons but does not validate uniqueness, stability, or causal necessity under systematic diagnosis.
- **Intervention baselines** are often omitted: showing that ablating a neuron changes behavior does not establish that it is the minimal or unique such neuron.

---

## Core Method

**Five-Step Diagnostic Protocol.** The paper proposes a minimum standard for sparse-neuron localization claims:

1. **Feature Correlation Analysis:** Compute pairwise Pearson correlation among features; flag selected neurons with |r| > 0.7 with other features as potentially non-unique.
2. **Bootstrap Stability:** Re-run neuron selection on bootstrap samples of the probe training data; measure how consistently the same neurons are selected.
3. **Sparse vs. Dense Ranking Disagreement:** Compare the ranking of neurons by the sparse probe with their ranking by a dense (non-regularized) probe or importance score; high disagreement signals that L1 regularization is driving the selection, not neuron importance per se.
4. **Intervention Baselines:** Compare targeted neuron ablation with ablation of randomly selected, equally predictive neurons; a uniquely causal neuron should show strictly greater behavior change.
5. **Cross-Dataset Evaluation:** Test selected neurons on held-out datasets not used for selection; poor transfer indicates dataset-specific selection rather than a general hallucination neuron.

**Application to H-Neurons.** The protocol is applied to H-Neurons in Gemma 3 4B and MedGemma 4B across TriviaQA, BioASQ, and NQ-Open.

---

## Technical Formulation

Let $\mathbf{h} \in \mathbb{R}^d$ be the activations at a given layer, and $y \in \{0,1\}$ be the hallucination label.

Sparse probing selects a neuron set $S$ by solving:
$$\hat{\beta} = \arg\min_\beta \;\mathcal{L}(y, \mathbf{h}^\top \beta) + \lambda \|\beta\|_1$$

The protocol flags $S$ as potentially non-unique when:
- $\exists j \notin S, i \in S$ such that $|\text{Corr}(h_i, h_j)| > 0.7$
- Bootstrap selection stability $< 0.8$ (Jaccard overlap across bootstrap runs)
- Spearman rank correlation between $\hat{\beta}$ and a dense-probe importance ranking $< 0.5$

Cross-dataset AUROC gap (Gemma 3 4B vs MedGemma 4B):
- TriviaQA: $+0.311$ vs $+0.235$
- BioASQ: $+0.474$ vs $+0.455$
- NQ-Open: $+0.128$ vs $+0.112$

---

## Learning or Inference Procedure

The diagnostic is applied post-hoc to trained probes:
1. Train an L1-regularized probe on each dataset.
2. Record the selected neuron set $S$.
3. Apply each diagnostic step in sequence.
4. Report pass/fail for each criterion and the overall diagnostic verdict.

No model training is performed; the protocol is a validation framework for existing neuron-selection claims.

---

## What the Paper Claims

Sparse predictive structure can coexist with non-unique neuron selection. Across three Gemma 3 4B settings:
- 19 of 22 selected H-Neurons have Pearson |r| > 0.7 with other features (high correlation, potential non-uniqueness).
- Bootstrap selections show only moderate stability.
- Sparse and dense rankings overlap only weakly.

This does not falsify the existence of neurons predictive of hallucination; it shows that the selected neurons are not the unique causal locus claimed.

---

## Experimental Findings

- Evaluated Gemma 3 4B and MedGemma 4B (a medically specialized variant).
- Gemma 3 4B consistently outperforms MedGemma 4B on hallucination detection across all three datasets.
- Diagnostic shows 19/22 H-Neurons fail the feature correlation criterion.
- Bootstrap Jaccard overlap is moderate, indicating sensitivity to training subsample.
- Sparse-dense rank correlation is weak, indicating L1 regularization drives neuron selection.

---

## Ablations and Interpretation

- **AUROC vs. uniqueness:** High AUROC is achievable by many overlapping neuron sets; predictive success does not imply localization.
- **Dataset specificity:** Neuron selection differs across TriviaQA, BioASQ, and NQ-Open, consistent with dataset-specific rather than universal hallucination neurons.
- **Implications for steering:** Interventions on selected neurons may or may not generalize because many correlated alternatives exist; robust steering requires either denser probes or causal structure beyond predictive accuracy.

---

## Reference

**Hallucination Neurons and Where to Find Them: An Investigation into the Existence of Hallucination Neurons**. arXiv:2609.29781, September 2026.  
https://arxiv.org/abs/2609.29781
