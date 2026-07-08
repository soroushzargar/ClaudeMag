# Online Conformal Prediction via Universal Portfolio Algorithms

**Authors:** Tuo Liu, Edgar Dobriban, Francesco Orabona
**Affiliation:** University of Pennsylvania, Boston University
**Venue:** ICML 2026 Spotlight
**arXiv:** 2602.03168
**URL:** https://arxiv.org/abs/2602.03168

---

## Headline Finding

Online conformal prediction promises long-run coverage even on adversarial data streams, but every prior method needed a hand-tuned learning rate to actually deliver it. This paper eliminates the tuning entirely by reducing the problem to portfolio selection and solving it with universal portfolio algorithms — the resulting method, UP-OCP, is parameter-free by construction.

---

## Key Findings (Inverted Pyramid)

### Level 1 — Main Result
UP-OCP achieves long-run $(1-\alpha)$ coverage on arbitrary, possibly adversarial data streams without any learning-rate tuning, by mapping the interval-calibration update to a two-asset portfolio-selection problem and solving it with a universal portfolio algorithm.

### Level 2 — Core Mechanism
Existing online conformal prediction (OCP) methods, most descending from Adaptive Conformal Inference (ACI), update a coverage threshold at each step using something like online gradient descent on the pinball loss. Their coverage guarantees hold only for a specific, hand-picked learning rate — get it wrong, and coverage can drift badly off target on hard or adversarial streams. This paper instead asks: what property of an online algorithm's *regret* actually implies a coverage guarantee, for any algorithm at all?

### Level 3 — Methodology
The authors develop a general regret-to-coverage theory for interval-valued OCP built on the $(1-\alpha)$-pinball loss. Their key structural insight is that controlling a specific notion of *linearized regret* — rather than the loss itself — implies long-run coverage bounds for any online algorithm, via a black-box reduction that depends only on the Fenchel conjugate of an upper bound on the linearized regret. This decouples "does it cover" from "which algorithm produced the threshold." They then recognize that the threshold-update problem, viewed through this regret lens, is structurally identical to sequential portfolio selection between two assets, letting them import decades of no-tuning-required universal portfolio algorithms (in the tradition of Cover's Universal Portfolio) directly into the conformal setting.

### Level 4 — Technical Details
Let $s_t$ be the nonconformity score at time $t$ and $q_t$ the calibrated threshold, with pinball loss
$$
\rho_{1-\alpha}(u) = \begin{cases} (1-\alpha)\,u & u \ge 0 \\ -\alpha\,u & u < 0 \end{cases}, \quad u = q_t - s_t.
$$
Define regret $R_T = \sum_{t=1}^T \rho_{1-\alpha}(q_t - s_t) - \min_q \sum_{t=1}^T \rho_{1-\alpha}(q - s_t)$. The paper shows the empirical coverage error is controlled by $R_T$ through a bound derived from the Fenchel conjugate of the regret's upper envelope. Casting $q_t$'s update as a bet fraction between a "covered" and "not covered" asset reduces the problem to constant-rebalanced portfolio selection, for which universal portfolio algorithms guarantee $O(\log T)$ regret with no learning-rate parameter — which, through the regret-to-coverage bound, translates directly into a parameter-free coverage guarantee.

---

## Why Existing Approaches Fall Short

ACI-style OCP methods require a learning rate that trades off responsiveness (tracking distribution shift quickly) against stability (not overreacting to noise), and the right choice is stream-dependent and unknowable in advance — exactly the adversarial setting OCP is meant to handle without such foreknowledge. Prior fixes required algorithm-specific regret analyses redone from scratch for each new update rule. The regret-to-coverage reduction here is algorithm-agnostic: any method with the right regret bound inherits a coverage guarantee automatically, and picking a universal portfolio algorithm removes the tuning problem altogether rather than just making it more forgiving.

---

## Experimental Findings

On time-series and deliberately adversarial data streams, UP-OCP matches or exceeds the empirical coverage of ACI variants that required per-stream learning-rate tuning, while UP-OCP itself uses none — removing an entire axis of hyperparameter search from online conformal deployment.

---

## Significance

Learning-rate tuning is the practical Achilles' heel of online conformal methods: it is exactly the kind of choice that can't be validated in the adversarial or non-stationary settings OCP exists for. By showing that coverage follows from a general regret bound rather than a specific algorithm, and that the resulting regret-minimization problem is literally a portfolio-selection problem with known parameter-free solutions, this paper turns online conformal prediction from a tuning exercise into a reduction — a template likely to extend well past the two-asset interval case studied here.
