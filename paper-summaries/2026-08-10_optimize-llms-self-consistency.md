# Position: It's Time to Optimize LLMs for Self-Consistency

**arXiv:** 2608.05188
**Authors:** Itamar Pres, Belinda Z. Li, Laura Ruis, Zifan Carl Guo, Keya Hu, Mehul Damani,
             Isha Puri, Ekdeep Singh Lubana, Jacob Andreas
**Affiliation:** MIT CSAIL and other institutions
**Venue:** arXiv preprint (position paper)
**Date:** July 31 / August 2026
**URL:** https://arxiv.org/abs/2608.05188

---

## Summary

This position paper argues that many persistent LLM failures—sycophancy, incomplete logical generalization, and overconfident wrong answers—share a common root cause: training and evaluation treat behavior as assessable on single input-output pairs, ignoring cross-input consistency relationships. The authors show that diverse existing techniques (paraphrase robustness training, logical consistency regularization, calibration) are special cases of a unified "consistency optimization" framework, and argue the field should explicitly adopt self-consistency as a first-class training objective.

## The Core Argument

Modern LLMs are trained to maximize the log-likelihood of individual (input, output) pairs. This point-wise training signal is inherently blind to **relational properties** of a model's behavior:
- Does the model give the same answer when a question is rephrased?
- Does it apply a logical rule it can state to a case where the rule applies?
- Does it maintain a claim under follow-up pressure?

A model can score perfectly on each individual example while being wildly inconsistent across them. Current RLHF and SFT pipelines do not directly penalize such inconsistency, because the reward or loss is computed over one (prompt, response) pair at a time.

## Three Failure Modes, One Root Cause

**Sycophancy:** LLMs change their stated answers when the user expresses disagreement, even when the model's original answer was correct. Point-wise training with human preference labels creates pressure to always agree with the annotator, without penalizing the inconsistency of agreeing when correct and also agreeing when the user is wrong.

**Incomplete logical generalization:** A model can correctly state a logical rule ("if P then Q") and correctly evaluate specific instances of P, yet fail to apply the rule in novel configurations. Point-wise evaluation checks each capability separately; cross-instance consistency checks reveal the gap.

**Overconfident wrong answers:** A model may give high-confidence incorrect answers on one framing of a problem while giving the correct answer on a differently framed version. This inconsistency is invisible to accuracy metrics measured per-question.

## Self-Consistency as a Unifying Framework

The paper observes that many existing "fixes" for these failures are implicitly consistency-based:
- **Data augmentation with paraphrases**: teaches the model to give the same answer to paraphrases → consistency across rephrasing
- **Logical consistency regularization (e.g., LoGIC)**: penalizes contradicting logical entailments → consistency across deductive steps
- **Calibration objectives**: align stated confidence with empirical accuracy across question sets → consistency between confidence and correctness
- **Self-play and debate**: model must defend claims against a skeptic → consistency under pressure

The authors formalize this as **consistency optimization**: a training objective that penalizes discrepancies in model behavior across inputs that should produce consistent outputs:

$$\mathcal{L}_{\mathrm{cons}}(\theta) = \mathbb{E}_{(x, x') \sim \mathcal{R}}\!\left[d\!\left(f_\theta(x),\, f_\theta(x')\right)\right]$$

where $\mathcal{R}$ is a relation specifying which input pairs $(x, x')$ should yield consistent outputs (paraphrases, logical implications, confidence-matched sets) and $d$ is an appropriate divergence.

## The Call to Action

The paper argues that:
1. Self-consistency should be a **first-class evaluation metric** alongside accuracy, not a secondary diagnostic
2. Consistency optimization should be incorporated **explicitly** into training pipelines (RLHF, SFT) rather than addressed implicitly through ad-hoc patches
3. The research community should build **consistency benchmarks** that expose cross-input behavior, not just per-instance performance

## Why It Matters

As LLMs are deployed in high-stakes settings (medicine, law, education), inconsistency is not a theoretical concern but a practical failure mode. A medical AI that gives different diagnoses when a patient's case is phrased differently is unsafe regardless of its average diagnostic accuracy. Framing self-consistency as a unified training objective—rather than a zoo of separate fixes—provides a clear research agenda and a common evaluation standard.

## Key Contributions

1. Unifying characterization of sycophancy, logical incompleteness, and overconfidence as consistency failures
2. Formal framework showing that diverse existing techniques are special cases of consistency optimization
3. Proposal for consistency-based evaluation metrics and training objectives
4. Survey of evidence that point-wise optimization leaves persistent consistency gaps
