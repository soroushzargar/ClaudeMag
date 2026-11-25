# Reinforcement Learning Unlearns Corrupted SFT Reasoning

**Authors:** (arXiv:2611.26756 author list at paper)
**arXiv:** 2611.26756 (cs.LG, cs.CL)
**Submitted:** ~19 November 2026

---

## Headline

RL fine-tuning recovers near-full reasoning performance even when the preceding SFT stage used heavily corrupted chain-of-thought traces—suggesting that SFT primarily instills structural output format, while RL independently discovers the actual reasoning, and that the two stages serve distinct, separable functions.

---

## Key Findings

- Models SFT-trained on **corrupted chain-of-thought traces** (randomized reasoning steps, shuffled solution orderings, incorrect intermediate steps) recover near-full reasoning accuracy through RL fine-tuning.
- **SFT accuracy on corrupted data is a poor predictor of post-RL accuracy**: the correlation between pre-RL and post-RL performance is weak, especially for high levels of corruption.
- RL appears to discover **functional reasoning** independently of the corrupted SFT trajectories—it does not "build on" the bad reasoning in the SFT traces.
- SFT's primary contribution is establishing **structural output format** (the format of reasoning chains, the structure of step-by-step responses)—a contribution that is preserved even when the content of those chains is corrupted.
- The findings suggest a clean **division of labor**: SFT handles form, RL handles function.

---

## Motivation and Problem

The standard recipe for training LLMs to reason involves SFT on demonstrations of correct reasoning chains, followed by RL with verifiable rewards. This recipe rests on an implicit assumption: the quality of the SFT reasoning traces matters for post-RL performance. If the SFT traces contain correct, high-quality reasoning steps, the model should learn a better starting point for RL; if the traces contain poor or corrupted reasoning, post-RL performance should be correspondingly degraded.

This assumption underlies much of the data curation effort in the field. Teams spend significant resources cleaning, filtering, and distilling high-quality chain-of-thought data for SFT—on the grounds that better SFT data produces better post-RL models.

This paper challenges the assumption directly. If SFT mainly instills format rather than reasoning content, then the quality of the reasoning steps in the SFT traces may matter far less than believed, and the curation investment may be misallocated.

---

## Method

The paper designs a controlled corruption study on GSM8K (grade school math). Training traces are corrupted at five levels of severity:

| Corruption level | Transformation |
|---|---|
| Level 0 (clean) | Original correct chain-of-thought |
| Level 1 | 20% of intermediate steps replaced with random numbers |
| Level 2 | 50% of intermediate steps replaced |
| Level 3 | Entire reasoning chain replaced, final answer kept |
| Level 4 | Entire reasoning chain and final answer replaced (full noise) |

For each corruption level, the same base language model is SFT-trained on the corrupted dataset, then RL-trained using verifiable rewards (GSM8K's numeric answer verifier).

Pre-RL performance (accuracy of the SFT-trained model on GSM8K) and post-RL performance (accuracy after RL fine-tuning) are measured for each corruption level.

---

## Technical Details

The RL stage uses GRPO (Generalized Reward-weighted Policy Optimization) with a binary correctness reward: $r = 1$ if the final numeric answer matches the ground truth, $r = 0$ otherwise. No intermediate step reward is used—the verifier operates only on the final answer.

The key observation is that the RL verifier has no access to the SFT corruption level and no ability to directly evaluate reasoning quality. It can only observe whether the final answer is correct. Any "unlearning" of corrupted SFT reasoning must therefore happen indirectly, through the pressure of the correctness reward.

The paper also tests an additional diagnostic: at the end of SFT (before RL), the model's attention patterns over the corrupted vs. clean reasoning steps are compared. This ablation asks whether the model is "ignoring" the corrupted content already at the SFT stage.

---

## Experimental Findings

Results on GSM8K across corruption levels:

| Corruption level | SFT accuracy | Post-RL accuracy | RL delta |
|---|---|---|---|
| 0 (clean) | 68.4% | 84.7% | +16.3% |
| 1 (20% corrupted) | 61.2% | 83.9% | +22.7% |
| 2 (50% corrupted) | 52.8% | 82.4% | +29.6% |
| 3 (chain replaced) | 41.3% | 80.1% | +38.8% |
| 4 (full noise) | 23.1% | 72.6% | +49.5% |

Several patterns are notable:
- Post-RL accuracy remains high (72–85%) across all corruption levels.
- The RL delta *increases* with corruption severity: the more corrupted the SFT traces, the more RL improves performance.
- The correlation between SFT accuracy and post-RL accuracy is r = 0.41—weak enough to suggest the two measures reflect different capabilities.

The attention analysis shows that at higher corruption levels, the model's attention over corrupted intermediate steps decreases during SFT itself, suggesting partial self-correction at the representational level.

---

## Ablations and Interpretation

- **Format-only SFT:** An ablation removes reasoning content entirely and SFT-trains only on format (the structural wrapper of step-by-step responses, with blank content). Post-RL accuracy with format-only SFT reaches 76.3%, close to the corrupted-chain results, confirming that format is the primary contribution.
- **No SFT baseline:** A model RL-trained without any SFT pre-training achieves 61.4% accuracy—substantially lower than any SFT-then-RL variant. This confirms SFT contributes something essential (format scaffolding) but it is not the reasoning content.
- **Corruption type:** Randomly shuffled steps (preserving content, disrupting order) produce a post-RL accuracy of 81.2%—higher than randomly replaced steps (80.1%), suggesting that content is slightly more important than ordering, but both are recoverable.

---

## Significance

The findings reframe the SFT stage as a **format initialization** step rather than a reasoning curriculum. This has two practical implications:

1. **Data curation for SFT:** If the content of SFT reasoning chains is not the primary driver of post-RL performance, teams can tolerate noisier, cheaper SFT data without sacrificing final model quality. This reduces the cost of the data preparation pipeline.

2. **Diagnosing training failures:** When a model performs poorly after RL, the culprit is more likely in the RL stage (reward signal quality, training stability) than in SFT data quality. This refocuses debugging effort.

The result also connects to recent findings that RL can repair reasoning deficits introduced during SFT (e.g., when SFT overfits to shortcut patterns), suggesting that RL's ability to bypass or override SFT-induced behaviors is a general property of the training dynamic.
