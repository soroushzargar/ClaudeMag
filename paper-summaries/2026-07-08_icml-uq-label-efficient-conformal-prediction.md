# Provably Label-Efficient Conformal Prediction

**Authors:** Andrew Ilyas, Joonhyuk Ko, Jingwu Tang, Zhiwei Steven Wu, Jiahao Zhang
**Affiliation:** Massachusetts Institute of Technology, Carnegie Mellon University
**Venue:** ICML 2026 Poster
**URL:** https://icml.cc/virtual/2026/poster/65796

---

## Headline Finding

Conformal prediction's finite-sample coverage guarantee is usually presented as free once you have a calibration set — but collecting those labels often isn't free at all. This paper treats calibration labels as a costly, queryable resource and designs an online stopping rule that automatically balances labeling cost against prediction-set efficiency, cutting labeling cost by 41.4% ± 2.3% while preserving the exact coverage guarantee.

---

## Key Findings (Inverted Pyramid)

### Level 1 — Main Result
An online stopping rule for sequential label acquisition matches the best *fixed* calibration-set size in hindsight while exactly preserving conformal coverage, and empirically reduces total labeling cost by 41.4% ± 2.3% relative to using a fixed calibration budget chosen without this guidance.

### Level 2 — Core Mechanism
Split conformal prediction fixes a calibration set of size $n$ up front; more calibration data generally tightens prediction sets, but with diminishing returns, and every additional label has a real acquisition cost (annotator time, expert review, sensor cost). Existing conformal theory is silent on how many labels to collect — it assumes $n$ is given. This paper asks the practical question directly: given unlabeled examples arriving one at a time and the ability to query a label at a cost, when should you stop calibrating and start predicting?

### Level 3 — Methodology
The authors formalize a total-cost objective combining the cumulative cost of labels queried so far with the expected inefficiency (prediction-set size) of the conformal predictor that would result from stopping now, and design a sequential stopping rule that halts querying once the estimated marginal benefit of one more label — in expected set-size reduction — no longer exceeds its cost. The delicate technical point is that stopping *adaptively*, based on the data seen so far, can break the exchangeability that ordinary conformal guarantees depend on; the authors show the stopping rule can be constructed so the resulting calibration set still yields a conformal predictor with exact finite-sample coverage $1-\alpha$, regardless of the adaptive stopping decision.

### Level 4 — Technical Details
With per-label cost $c$ and a weight $\lambda$ trading off cost against expected prediction-set size $\mathbb{E}[|C(X)|]$, define the total objective after $m$ queried labels as
$$
J(m) = m \cdot c + \lambda \cdot \mathbb{E}\big[|C_m(X)|\big],
$$
where $C_m$ is the conformal predictor calibrated on the first $m$ labels. The stopping time is
$$
\tau = \inf\{m : \widehat{\Delta}(m) \le 0\},
$$
where $\widehat{\Delta}(m)$ estimates the marginal value of querying label $m+1$ (the expected reduction in $\lambda \cdot \mathbb{E}[|C(X)|]$ minus $c$). The authors prove that under mild regularity conditions, the expected total cost $\mathbb{E}[J(\tau)]$ matches $\min_m J(m)$ — the best *fixed* calibration size chosen with hindsight knowledge — up to lower-order terms, while the coverage guarantee $P(Y \in C_\tau(X)) \ge 1-\alpha$ holds exactly despite $\tau$ being a data-dependent stopping time.

---

## Why Existing Approaches Fall Short

Practitioners deploying conformal prediction today either guess a calibration-set size in advance (risking either wasted labeling budget or an under-calibrated, oversized prediction set) or collect labels until budget runs out with no statistical link between that stopping point and the tradeoff they actually care about. Naively stopping "when the sets look tight enough" — an adaptive, data-dependent rule — would ordinarily void the exchangeability conformal prediction relies on; this paper is the first to show a stopping rule can be both label-efficient and provably coverage-preserving at the same time.

---

## Experimental Findings

Across benchmark classification and regression tasks, the proposed stopping rule reduces total labeling cost by 41.4% ± 2.3% relative to fixed calibration-set-size baselines chosen without this guidance, while maintaining nominal coverage throughout.

---

## Significance

Calibration data is treated as a free resource throughout most of the conformal prediction literature, which quietly assumes it away as a solved logistics problem. In practice — medical labeling, expert annotation, expensive simulation — it is often the dominant cost of deploying a conformal system. By proving that adaptive, cost-aware stopping can coexist with an exact coverage guarantee, this paper turns "how much calibration data do I actually need" from a guess into a provable, optimizable question.
