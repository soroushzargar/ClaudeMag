# Calibrating Decision Robustness via Inverse Conformal Risk Control

**Authors:** Wenbin Zhou, Shixiang Zhu
**Affiliation:** Carnegie Mellon University
**Venue:** ICML 2026 Poster
**arXiv:** 2510.07750
**URL:** https://arxiv.org/abs/2510.07750

---

## Headline Finding

Robust optimization protects decisions by hedging against a conformal uncertainty set, but the robustness level behind that set is normally fixed ad hoc — too loose and decisions are exposed, too tight and they're needlessly conservative. Inverse Conformal Risk Control (ICRC) flips the usual recipe, delivering simultaneous, distribution-free guarantees on both miscoverage and downstream regret so a decision-maker can choose their own robustness level directly off a certified Pareto frontier.

---

## Key Findings (Inverted Pyramid)

### Level 1 — Main Result
ICRC constructs valid, finite-sample estimators for both miscoverage and optimization regret across an entire family of robust predict-then-optimize policies, tracing out the miscoverage-regret Pareto frontier rather than committing to a single a priori coverage target.

### Level 2 — Core Mechanism
In robust predict-then-optimize, an uncertainty set $\mathcal{U}_\alpha(x)$ is built via conformal calibration at level $\alpha$, and a decision is chosen to minimize the worst case cost over that set. The choice of $\alpha$ determines both how much protection the decision gets and how much optimization performance ("regret" relative to knowing the true outcome) is sacrificed for that protection — but conventional practice picks $\alpha$ once, in advance, with no principled way to know whether that choice was too cautious or not cautious enough for the task's actual cost structure.

### Level 3 — Methodology
Rather than treating $\alpha$ as a knob to be set once, the authors treat the *entire curve* relating miscoverage to regret as the object of interest, and ask for valid, simultaneous statistical guarantees on both quantities across a range of robustness levels — not just at one chosen in advance. This means constructing a family of policies indexed by robustness level, and providing distribution-free finite-sample bounds on both the true miscoverage rate and the true regret at every level in that family simultaneously, so a decision-maker can inspect the empirical Pareto frontier after the fact and select whichever point matches their actual cost-risk preference, with the guarantee still holding for that a posteriori choice.

### Level 4 — Technical Details
For a family of robust policies $\{\pi_\alpha\}$ built from uncertainty sets $\mathcal{U}_\alpha(x)$, define
$$
\text{miscoverage}(\alpha) = P\big(y \notin \mathcal{U}_\alpha(x)\big), \qquad
\text{regret}(\alpha) = \mathbb{E}\big[\text{cost}(\pi_\alpha(x), y)\big] - \min_{a} \mathbb{E}[\text{cost}(a, y)].
$$
These move in opposite directions as $\alpha$ varies, forming a Pareto frontier. ICRC constructs simultaneous confidence bands for both functions, valid uniformly over a grid of $\alpha$ values with probability at least $1-\delta$, using conformal calibration on a held-out set combined with uniform concentration arguments (in the style of DKW-type bounds) to control the multiple-testing-like inflation that would otherwise come from evaluating many $\alpha$ values on the same data. The decision-maker then selects $\hat{\alpha}$ off the *certified* empirical frontier, inheriting the finite-sample validity of both bounds at that selected point.

---

## Why Existing Approaches Fall Short

Fixing a single coverage target $\alpha$ in advance and only afterward measuring the regret it induces means the decision-maker learns whether their choice was good only after committing to it — and re-running the whole pipeline for a different $\alpha$ offers no statistical guarantee that the new choice is valid, since the guarantee was calibrated for the original $\alpha$ alone. ICRC's simultaneous validity across the whole frontier removes this trap: practitioners can compare robustness levels honestly, with statistical validity, before committing.

---

## Experimental Findings

On robust predict-then-optimize benchmarks, ICRC's certified Pareto frontier lets practitioners identify robustness levels that achieve materially lower regret than conventional fixed-$\alpha$ conformal robust optimization at matched miscoverage, and vice versa — demonstrating that the ad hoc $\alpha$ choices baked into prior robust-optimization pipelines were leaving performance on the table without the practitioner ever being able to see it.

---

## Significance

This paper reframes what "calibration" should mean in robust decision-making: not picking one coverage level and hoping it's the right one, but making the entire tradeoff between protection and performance statistically visible and choosable. As conformal uncertainty sets increasingly feed downstream optimization pipelines in operations, finance, and safety-critical planning, ICRC's Pareto-frontier view is a more honest and more useful object to hand to a decision-maker than a single, unexamined robustness knob.
