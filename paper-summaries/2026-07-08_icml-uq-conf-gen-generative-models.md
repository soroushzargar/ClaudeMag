# Conf-Gen: Conformal Uncertainty Quantification for Generative Models

**Authors:** Gabriel Loaiza-Ganem, Kevin Zhang, Wei Cui, Marc Law, Kin Kwan Leung
**Affiliation:** Layer 6 AI (TD Bank Group)
**Venue:** ICML 2026 Poster
**arXiv:** 2605.28920
**URL:** https://arxiv.org/abs/2605.28920

---

## Headline Finding

Conformal risk control gives formal guarantees for supervised predictors, but it has no obvious foothold in unsupervised generative systems — there is no "true label" to a diffusion sample or an LLM's answer. Conf-Gen relaxes CRC's assumptions enough to reach generative tasks directly, producing formal guarantees like "this image is not a memorized copy" or "this agent's output is correct."

---

## Key Findings (Inverted Pyramid)

### Level 1 — Main Result
Conf-Gen is a general framework that adapts conformal risk control (CRC) to generative tasks, unifying and generalizing prior ad hoc attempts to apply conformal prediction to LLMs, and extending the methodology to entirely new domains: non-memorization guarantees for image generators, sufficiency-of-clarification guarantees for conversational agents, and correctness guarantees for AI agent outputs.

### Level 2 — Core Mechanism
Standard CRC calibrates a threshold $\lambda$ against a monotone, bounded loss so that the expected risk stays below a target level $\alpha$, using an exchangeable sequence of labeled calibration pairs $(x_i, y_i)$. Generative models break this recipe at the root: a large language model or an image generator does not produce a prediction against a ground-truth label in the classical sense, so there is no natural (x, y) pair to calibrate against. Conf-Gen's contribution is to identify which of CRC's assumptions are load-bearing and which can be relaxed, replacing the labeled-pair requirement with task-specific risk functionals defined directly over generated outputs (e.g., distance-to-training-set as a memorization score, or a verifier's judgment as a correctness score) that remain calibratable under the same finite-sample machinery.

### Level 3 — Methodology
For each application, Conf-Gen defines a risk functional $\ell(\lambda, x)$ that is monotone in the threshold $\lambda$ (e.g., raising $\lambda$ makes the generation procedure more conservative, which lowers the risk of memorization or hallucination but potentially raises cost or reduces informativeness). Calibration proceeds by finding the smallest $\lambda$ for which the empirical risk, corrected for finite-sample slack, does not exceed $\alpha$, exactly mirroring CRC's calibration recipe — but with the object of calibration re-derived so that it applies to a single generative process rather than a labeled prediction task. This lets Conf-Gen reuse CRC's finite-sample guarantee machinery while targeting risk notions (memorization, question-sufficiency, correctness) that have no analogue in classical supervised conformal prediction.

### Level 4 — Technical Details
Let $R(\lambda) = \mathbb{E}[\ell(\lambda, X)]$ be the population risk as a function of threshold $\lambda$, for a monotone bounded loss $\ell$. Calibration selects
$$
\hat{\lambda} = \inf\left\{\lambda : \hat{R}_n(\lambda) + c_n(\delta) \le \alpha\right\},
$$
where $\hat{R}_n$ is the empirical risk over $n$ calibration instances and $c_n(\delta)$ is a finite-sample correction term ensuring $R(\hat{\lambda}) \le \alpha$ with probability at least $1-\delta$, in the spirit of Angelopoulos et al.-style CRC bounds. Conf-Gen's relaxation replaces the requirement that $(X, Y)$ be an exchangeable labeled pair with the weaker requirement that the sequence of *generation instances* (and their associated risk scores) be exchangeable — sufficient to preserve the coverage-style guarantee while accommodating memorization scores, clarification-sufficiency scores, and agent-correctness scores as the risk functional.

---

## Why Existing Approaches Fall Short

Prior attempts to bring conformal guarantees to LLMs typically bolted a supervised-style calibration step onto a generative pipeline in a task-specific, one-off way — for instance treating a single decoding step as if it were a classification label. These approaches don't generalize: each new generative task (image synthesis, dialogue, agentic tool use) required its own bespoke reformulation, with no shared theoretical foundation guaranteeing correctness. Conf-Gen instead identifies the general relaxation of CRC's assumptions that makes the whole class of problems tractable at once.

---

## Experimental Findings

The paper demonstrates Conf-Gen across three qualitatively different generative settings: certifying that image generator outputs are not memorized copies of training data, certifying that a conversational AI has asked sufficient clarifying questions before committing to an answer, and certifying the correctness of an AI agent's output — showing the same calibration recipe transfers across modalities and task types without redesign.

---

## Significance

As generative and agentic systems displace classical supervised predictors as the dominant AI paradigm, the uncertainty-quantification toolkit built for supervised learning risks becoming obsolete exactly when it's needed most. Conf-Gen is a bid to keep conformal guarantees — arguably the most practically trusted form of formal ML guarantee — relevant for the systems that matter now: LLMs, generators, and autonomous agents, rather than leaving that ground to informal or unverified confidence heuristics.
