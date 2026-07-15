# Nemotron-Labs-Diffusion: A Tri-Mode Language Model Unifying Autoregressive, Diffusion, and Self-Speculation Decoding

**arXiv:** 2607.05722  
**Date:** July 7, 2026  
**Authors:** Yonggan Fu, Lexington Whalen, Abhinav Garg, Chengyue Wu, Maksim Khadkevich, Nicolai Oswald, Enze Xie, Daniel Egert, Sharath Turuvekere Sreenivas, Shizhe Diao, Chenhan Yu, Ye Yu, Weijia Chen, Sajad Norouzi, Jingyu Liu, Shiyi Lan, Ligeng Zhu, Jin Wang, Jindong Jiang, Morteza Mardani, Mehran Maghoumi, Song Han, Ante Jukić, Nima Tajbakhsh, Jan Kautz, Pavlo Molchanov  
**Affiliation:** NVIDIA Research  
**URL:** https://arxiv.org/abs/2607.05722

---

## Headline

NVIDIA's Nemotron-Labs-Diffusion unifies autoregressive and diffusion language modeling into a single architecture that self-speculates — eliminating the separate draft model — to deliver 6x higher throughput than comparable autoregressive-only baselines.

---

## Key Findings

- A single set of weights can simultaneously support AR decoding, parallel diffusion decoding, and self-speculative decoding without any weight changes at inference time.
- In self-speculation mode, the diffusion head drafts candidate tokens and the AR head verifies them, achieving 6.82 accepted tokens per step versus Eagle3's 2.75.
- The AR and diffusion objectives are complementary: diffusion improves lookahead planning while AR provides left-to-right linguistic priors.
- Nemotron-Labs-Diffusion-8B achieves 6x more decoded tokens per forward pass than Qwen3-8B and 4x higher throughput on SPEED-Bench with SGLang on a GB200 GPU.
- Open weights are released in 3B, 8B, and 14B sizes under a commercial-permissive license.

---

## Methodology

The model is trained with a joint AR-diffusion objective on the same token sequence. The diffusion training arm masks a random span of tokens and trains the model to reconstruct them conditioned on unmasked context, while the AR arm trains left-to-right prediction. At inference, a routing mechanism selects the decoding mode:

1. **AR mode:** Standard left-to-right autoregressive generation, used when latency per token matters most.
2. **Diffusion mode:** A parallel sampling loop denoises a block of masked tokens simultaneously, then applies an optimized sampler trained on sampling trajectories.
3. **Self-speculation mode:** The diffusion head proposes a draft block; the AR head scores each draft token in parallel and accepts/rejects per the speculative decoding acceptance criterion.

The key architectural insight is that the diffusion head naturally functions as a draft model because its parallel generation is cheap, while the AR head naturally functions as a verifier because it has strong left-to-right priors.

---

## Technical Details

- **Training objective:** $\mathcal{L} = \alpha \mathcal{L}_{AR} + (1-\alpha)\mathcal{L}_{diffusion}$, where the diffusion loss is the masked token reconstruction cross-entropy at various mask rates.
- **Draft acceptance:** Standard speculative decoding acceptance criterion — a draft token $x_d$ with AR probability $p_{AR}(x_d)$ and draft probability $p_D(x_d)$ is accepted with probability $\min(1, p_{AR}(x_d)/p_D(x_d))$.
- **Mask schedule:** The diffusion arm uses a variable masking ratio $\rho \sim \text{Beta}(a,b)$ that encourages the model to learn both local and long-range dependencies.
- **Sampler:** An additional lightweight MLP sampler is trained to optimize over diffusion sampling trajectories, reducing the number of diffusion steps required for high-quality drafts.

---

## Experimental Findings

- On SPEED-Bench (throughput-accuracy Pareto frontier), Nemotron-Labs-Diffusion-8B dominates Qwen3-8B, LLaMA-4-8B, and MTP-based models across all concurrency levels tested.
- Accepted tokens per step: Nemotron-Labs-Diffusion 6.82 vs. Eagle3 (separate draft model) 2.75.
- Quality (as measured by MT-Bench, MMLU, HumanEval) is within 1% of the autoregressive-only baseline at equivalent parameter counts.
- Mode switching overhead is negligible (<0.5% latency penalty) due to shared weights.
- The 3B model outperforms the 7B Eagle3 draft+verifier pair on throughput while using fewer total parameters.

---

## Ablations and Interpretation

- Removing the diffusion training arm causes the self-speculation acceptance rate to drop from 6.82 to 4.1 tokens per step, confirming that diffusion training is the source of the improved drafting quality.
- Increasing $\alpha$ (more AR weight) improves text quality at the cost of draft acceptance rate; $\alpha = 0.5$ is the Pareto-optimal point.
- Self-speculation outperforms multi-token prediction (MTP) because MTP trains the head to predict multiple tokens autoregressively, while diffusion trains the head to predict any masked subset, making it a more versatile drafter.

---

## Reference

Yonggan Fu et al. **Nemotron-Labs-Diffusion: A Tri-Mode Language Model Unifying Autoregressive, Diffusion, and Self-Speculation Decoding.** arXiv:2607.05722, July 2026. https://arxiv.org/abs/2607.05722
