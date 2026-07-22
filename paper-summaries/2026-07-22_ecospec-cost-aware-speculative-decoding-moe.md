# Less Experts, Faster Decoding: Cost-Aware Speculative Decoding for Mixture-of-Experts

**arXiv:** 2607.12696
**Date:** July 2026
**Authors:** Jincheng Xie, Runheng Liu, Heyan Huang, Yawen Ling, Hanbin Dai, Yu Zheng, Wen Hu
**Affiliation:** Beijing Institute of Technology
**URL:** https://arxiv.org/abs/2607.12696

---

## Headline

Researchers at Beijing Institute of Technology identify expert scattering — a previously uncharacterized failure mode of speculative decoding in Mixture-of-Experts models — and introduce EcoSpec, a cost-aware framework that co-optimizes draft acceptance and expert reuse, achieving up to 1.62x end-to-end decoding speedup on DeepSeek-V3.1, Qwen3-235B, and GPT-OSS-120B.

---

## Key Findings

- Expert scattering is a novel failure mode: standard speculative decoding selects draft tokens by confidence alone, but high-confidence tokens may route to disjoint expert sets, eliminating the memory-locality benefit that makes speculation fast in practice.
- EcoSpec introduces a lightweight expert predictor that forecasts which experts a draft token will activate before that token is committed, enabling cost-aware draft selection.
- A dynamic expert buffer tracks which experts are already loaded into GPU fast memory for the current verification batch, biasing draft selection toward tokens that reuse those experts.
- EcoSpec achieves up to 1.62x end-to-end speedup over standard speculative decoding baselines on large-scale MoE models, without degrading output quality (acceptance rates maintained or improved).
- The method is model-agnostic and requires no changes to model weights; it acts as a drop-in replacement for the draft selection strategy.

---

## Methodology

Standard speculative decoding for MoE models selects draft tokens greedily by likelihood under the draft model, then verifies them in a single forward pass through the target (large) model. EcoSpec modifies draft selection:

1. **Expert predictor:** A small auxiliary network (linear over the draft model's hidden states) predicts which of the target model's experts each candidate token will activate.
2. **Expert buffer:** A running set tracks which expert weights are currently cached in GPU high-bandwidth memory (HBM) for the active verification batch.
3. **Cost-aware scoring:** Each candidate draft token receives a composite score combining acceptance likelihood $p_{\text{accept}}$ and expert reuse fraction $f_{\text{reuse}}$:
   $$s_i = p_{\text{accept},i} \cdot (1 + \lambda \cdot f_{\text{reuse},i})$$
   where $\lambda$ is a hyperparameter balancing the two objectives.
4. **Dynamic buffer update:** After each verification step, the buffer is updated with the experts activated by the accepted token sequence, reflecting the current HBM state.

---

## Technical Details

- **Expert predictor training:** The predictor is trained offline on a small calibration corpus to minimize prediction error for expert activation patterns; it adds negligible inference latency (~0.3% of target model forward pass time).
- **Buffer size:** The expert buffer is sized to match the HBM capacity of the target serving setup; experiments use buffer sizes of 8–32 experts (out of 64 total for DeepSeek-V3.1 and Qwen3-235B-A22B).
- **Scoring function:** $\lambda = 0.4$ is used across all experiments; sensitivity analysis shows robustness to values in $[0.2, 0.8]$.
- **Speculative draft model:** A small draft model (typically 7B–13B parameters) generates candidate tokens; EcoSpec modifies only the token selection step, not the draft model itself.
- **Models evaluated:** DeepSeek-V3.1 (671B total, 37B active per token), Qwen3-235B-A22B (235B total, 22B active), GPT-OSS-120B (120B total, 20B active).

---

## Experimental Findings

| Model | Task | Standard SD | EcoSpec | Speedup |
|---|---|---|---|---|
| DeepSeek-V3.1 | Reasoning | 1.00x | 1.62x | +62% |
| DeepSeek-V3.1 | Coding | 1.00x | 1.51x | +51% |
| Qwen3-235B-A22B | QA | 1.00x | 1.44x | +44% |
| GPT-OSS-120B | Dialogue | 1.00x | 1.38x | +38% |

EcoSpec consistently reduces active expert footprint (measured as unique experts loaded per token batch) by 18–35% relative to confidence-only draft selection, without reducing the token acceptance rate.

---

## Ablations and Interpretation

- **Expert predictor vs. oracle:** Using a perfect expert predictor (oracle from ground truth) improves speedup to 1.81x for DeepSeek-V3.1, indicating the expert predictor quality is the primary bottleneck — room for improvement as predictor architectures advance.
- **Dynamic vs. static buffer:** A static expert buffer (fixed set, not updated each step) achieves only 1.22x speedup, confirming that tracking dynamic HBM state is essential.
- **Lambda sensitivity:** Speedup varies from 1.41x ($\lambda=0$, confidence only) to 1.62x ($\lambda=0.4$) to 1.48x ($\lambda=1.0$, reuse only), confirming that neither pure confidence nor pure reuse dominates and joint optimization is key.
- **Expert scattering baseline:** Without EcoSpec, standard speculative decoding achieves negative speedup on DeepSeek-V3.1 at high draft lengths ($\geq 8$ tokens), confirming expert scattering as a practical barrier for MoE speculative decoding at scale.

---

## Reference

Jincheng Xie, Runheng Liu, Heyan Huang, Yawen Ling, Hanbin Dai, Yu Zheng, Wen Hu. **Less Experts, Faster Decoding: Cost-Aware Speculative Decoding for Mixture-of-Experts.** arXiv:2607.12696, July 2026. https://arxiv.org/abs/2607.12696
