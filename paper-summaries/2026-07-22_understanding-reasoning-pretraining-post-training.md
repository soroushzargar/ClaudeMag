# Understanding Reasoning from Pretraining to Post-Training

**arXiv:** 2607.16097
**Date:** July 17, 2026
**Authors:** Jingyan Shen, Ang Li, Salman Rahman, Yifan Sun, Micah Goldblum, Matus Telgarsky, Pavel Izmailov
**Affiliation:** New York University, Modal Labs, UCLA, UIUC, Columbia University
**URL:** https://arxiv.org/abs/2607.16097

---

## Headline

Researchers from NYU and collaborators use chess as a controlled testbed to discover a joint pretraining-to-RL scaling law, revealing that RL post-training does not invent new capabilities but reorganizes and concentrates existing competence already encoded during pretraining.

---

## Key Findings

- A joint scaling law governs post-RL reward as a function of both pretraining loss and RL compute budget, enabling principled allocation of compute across the two training stages.
- RL post-training does not discover qualitatively new moves; it reorganizes existing competence by amplifying high-value moves and suppressing low-value ones that pretraining spread probability mass across.
- The marginal return to RL compute decreases faster when pretraining quality is lower, meaning RL is not a remedy for weak pretraining but a multiplier on strong pretraining.
- Chess provides a fully observable, ground-truth-evaluable environment where the policy can be probed move-by-move, yielding mechanistic insights that are opaque in language tasks.
- Models are pretrained from 5M to 1B parameters; the scaling law holds across all sizes studied.

---

## Methodology

The authors construct a clean experimental system using chess as the reasoning domain:

1. **Pretraining:** Language models (5M–1B parameters) are pretrained on corpora of human chess games in PGN notation, providing exposure to high-quality move distributions.
2. **Supervised fine-tuning:** Models are SFT'd on synthetically generated chain-of-thought reasoning traces that explain candidate moves in terms of board evaluation, threat assessment, and strategic planning.
3. **RL post-training:** Models are RL-trained on chess puzzles using outcome reward (correct solution: +1, incorrect: 0), analogous to math or code RL with verifiable answers.
4. **Mechanistic analysis:** Move-level probability distributions are compared across pretraining, SFT, and RL checkpoints to trace how each stage reshapes the policy.

---

## Technical Details

- **Joint scaling law:** Post-RL reward $R$ is expressed as $R(L_{\text{pre}}, C_{\text{RL}}) = R_\infty - \phi(L_{\text{pre}}) \cdot C_{\text{RL}}^{-\gamma}$, where $L_{\text{pre}}$ is pretraining loss, $C_{\text{RL}}$ is RL compute, $\phi$ is a monotone function of pretraining quality, and $\gamma$ is a universal exponent.
- **Models:** Decoder-only Transformers trained from scratch with standard next-token prediction on PGN game records sourced from Lichess.
- **RL algorithm:** GRPO with chess puzzle correctness as binary reward; no intermediate step rewards.
- **Code and models:** Released at github.com/pavelslab-nyu/pre2post-chess and huggingface.co/pavelslab-nyu/pre2post-chess.

---

## Experimental Findings

- The joint scaling law fits held-out (pretraining size, RL compute) combinations with low relative error across the full parameter range studied (5M to 1B).
- Under fixed total compute, the optimal allocation shifts toward more RL compute as pretraining quality improves — stronger pretrained models extract more value from RL.
- RL post-training achieves the largest absolute reward gains for models with the lowest pretraining loss (best pretrained models), confirming RL as a multiplier rather than a corrector.
- Move-level analysis shows RL concentrates probability mass on the top-1 move for positions where the pretrained model already had the correct move in its top-3 but spread probability across alternatives.

---

## Ablations and Interpretation

- **SFT vs. RL:** SFT on reasoning traces improves puzzle accuracy but plateaus quickly; RL continues improving beyond the SFT ceiling, reaching qualitatively higher accuracy.
- **RL without SFT:** RL applied directly to the pretrained model (no SFT) converges more slowly and to a lower final reward, confirming SFT as a useful warm-start for RL.
- **Scaling law extrapolation:** The fitted scaling law accurately predicts post-RL reward for a 1B-parameter model trained with 10x more RL compute than any training run used to fit it.
- **Mechanistic result:** The probability redistribution caused by RL is local — it sharpens distributions around moves already present, rather than assigning probability to moves that pretraining assigned near-zero probability to, suggesting RL cannot inject genuinely novel knowledge.

---

## Reference

Jingyan Shen, Ang Li, Salman Rahman, Yifan Sun, Micah Goldblum, Matus Telgarsky, Pavel Izmailov. **Understanding Reasoning from Pretraining to Post-Training.** arXiv:2607.16097, July 2026. https://arxiv.org/abs/2607.16097
