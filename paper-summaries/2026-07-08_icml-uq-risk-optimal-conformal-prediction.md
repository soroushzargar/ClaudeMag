# Optimal Decision-Making Based on Prediction Sets

**Authors:** Tao Wang, Edgar Dobriban
**Affiliation:** University of Pennsylvania (Wharton Statistics and Data Science)
**Venue:** ICML 2026 Oral
**arXiv:** 2602.00989
**URL:** https://arxiv.org/abs/2602.00989

---

## Headline Finding

Conformal prediction sets carry a coverage guarantee, but until now there was no principled answer to *what to do* with them once you have to make a decision. This paper derives the minimax-optimal downstream policy for a fixed prediction set, then the optimal prediction set itself, and packages both into a single practical algorithm: Risk-Optimal Conformal Prediction (ROCP).

---

## Key Findings (Inverted Pyramid)

### Level 1 — Main Result
ROCP consistently reduces worst-case risk and critical mistake rates relative to standard risk-averse max-min policies and naive best-response policies built on top of ordinary conformal sets — with the largest gains appearing exactly where errors outside the prediction set are most costly.

### Level 2 — Core Mechanism
A prediction set $C(x)$ only promises that the true label lands inside it with probability at least $1-\alpha$; it says nothing about which action to take given that set, nor whether the set itself is shaped well for the decision task downstream. The paper treats this as a decision-theoretic problem: find the policy that minimizes expected loss against the *worst-case* distribution still consistent with the coverage guarantee.

### Level 3 — Methodology
The authors first solve for the minimax-optimal policy for an arbitrary fixed set $C(x)$: it turns out to balance the worst-case loss for labels inside $C(x)$ against a penalty for the residual probability mass ($\alpha$) that could fall outside the set. Holding that policy fixed, they then solve backward for the *optimal set construction* — the set that minimizes the resulting robust risk subject to the marginal coverage constraint. ROCP operationalizes this two-step optimum: a black-box probabilistic model estimates the population quantities needed for the oracle decision rule, and a held-out calibration set converts the estimate into a finite-sample guarantee via standard conformal calibration.

### Level 4 — Technical Details
For decision $a \in \mathcal{A}$ and loss $\ell(a,y)$, define the robust risk of policy $\pi$ given set $C$ as
$$
R(\pi, C) = \sup_{Q:\, Q(y \in C(x)) \ge 1-\alpha} \mathbb{E}_Q[\ell(\pi(C(x)), y)].
$$
The minimax-optimal $\pi^*$ balances $\sup_{y \in C(x)} \ell(\pi^*, y)$ against an $\alpha$-weighted penalty for loss outside $C(x)$. The optimal set $C^*(x)$ then minimizes $R(\pi^*, C)$ subject to $P(y \in C(x)) \ge 1-\alpha$, and ROCP calibrates a nonconformity threshold on held-out data so that $C^*(x)$ retains exact finite-sample marginal coverage under exchangeability, regardless of model misspecification.

---

## Why Existing Approaches Fall Short

Prior work treated the conformal set as the end product and left "what to do with it" to ad hoc downstream heuristics — typically a naive best-response policy (act as if the point prediction were exact) or an overly conservative max-min policy (plan for the single worst label in the set, ignoring how likely it actually is). Neither is calibrated to the decision task: best-response ignores the coverage guarantee entirely, while max-min wastes performance hedging against labels the coverage guarantee doesn't actually make likely. ROCP is the first method to derive the policy and the set jointly from the same decision-theoretic objective.

---

## Experimental Findings

Across decision tasks where the cost of an out-of-set error is high, ROCP reduces both worst-case risk and the rate of critical (high-cost) mistakes compared to the max-min baseline (over-conservative) and the best-response baseline (under-protected), with the gap widening as the cost of out-of-set errors grows.

---

## Significance

This paper closes a gap that has quietly sat underneath the entire conformal prediction literature: coverage guarantees are not decision guarantees. By deriving the optimal decision policy and the optimal set construction from the same minimax objective — rather than bolting a heuristic decision rule onto a set built for coverage alone — ROCP gives conformal methods a principled bridge into the risk-sensitive decision-making settings (medical triage, financial risk limits, safety-critical control) where sets are ultimately meant to be acted on, not just reported.
