# When Has a Bayesian Neural Network Sampled Enough? Adaptive Inference Time with Statistical Guarantees

**arXiv:** 2610.12212  
**Authors:** Fabian Denoodt, Sibylle Hess  
**Venue:** NeurIPS 2026 Workshop (E-Values: From Statistics to ML)  
**Date:** October 8, 2026

---

## One-Line Summary

By replacing fixed-budget Monte Carlo sampling in Bayesian neural networks with anytime-valid confidence sequences, this paper terminates sampling per-input as soon as a statistical guarantee on the relevant decision is satisfied, concentrating compute on ambiguous inputs and reducing overall inference latency.

---

## Problem

Bayesian neural networks (BNNs) approximate the posterior predictive distribution $p(y \mid \mathbf{x}, \mathcal{D})$ by averaging over $S$ Monte Carlo samples from the weight posterior: $\hat{p}(y \mid \mathbf{x}) = \frac{1}{S}\sum_{s=1}^S p(y \mid \mathbf{x}, \mathbf{w}^{(s)})$. In practice $S$ is a fixed hyperparameter chosen before deployment, but this is wasteful: easy inputs (far from decision boundaries) are over-sampled, while hard inputs (near boundaries) may be under-sampled. Neither the decision reliability nor the latency is explicitly controlled.

---

## Why Existing Approaches Fall Short

- **Fixed-$S$ inference** ignores input difficulty; the same compute is used for obvious and ambiguous cases.
- **Early stopping heuristics** (e.g., stopping when the running mean stabilises) provide no formal guarantee on the error of the resulting estimate.
- **Conformal prediction** provides marginal coverage guarantees but applies to the full test set, not to individual-input sequential decisions.
- **Sequential hypothesis testing** (e.g., SPRT) was designed for binary tests, not for multi-class predictions or distributional estimation.

---

## Core Method

The paper introduces **confidence sequences** (CS) as the decision-making primitive for BNN inference. A confidence sequence $\{C_s\}_{s=1}^\infty$ is a sequence of sets such that the true parameter lies in $C_s$ for all $s$ simultaneously with probability at least $1-\alpha$:
$$\Pr\!\left[\exists s : \theta \notin C_s\right] \leq \alpha$$

This **anytime-valid** property means sampling can be stopped at any data-dependent time without inflating the error rate. The paper instantiates confidence sequences for three BNN inference settings:

1. **Argmax class:** Stop sampling when the confidence sequence for $p(y^* \mid \mathbf{x})$ excludes all alternatives.
2. **Full predictive distribution:** Stop when the confidence ellipsoid for the vector of class probabilities is smaller than a tolerance $\varepsilon$.
3. **Threshold decision:** Stop when the CS for $p(y \geq t \mid \mathbf{x})$ lies entirely above or below a decision threshold $\tau$.

---

## Technical Formulation

Let $\hat{p}_s^c = \frac{1}{s}\sum_{j=1}^s p(y=c \mid \mathbf{x}, \mathbf{w}^{(j)})$ be the running mean for class $c$.

A **KL-mixture** confidence sequence for the Bernoulli parameter $p^c$ gives a time-uniform interval $(L_s^c, U_s^c)$ satisfying:
$$\Pr\!\left[\exists s \geq 1 : p^c \notin (L_s^c, U_s^c)\right] \leq \alpha$$

The stopping rule for argmax decision is:
$$\tau^* = \min\!\left\{s : U_s^{c^*} > \max_{c \neq c^*} U_s^c\right\}$$
where $c^* = \arg\max_c \hat{p}_s^c$.

For distributional approximation, the confidence region is:
$$\Pr\!\left[\|\hat{\mathbf{p}}_s - \mathbf{p}\|_2 \leq r_s\right] \geq 1-\alpha \quad \forall s$$

with $r_s = O(\sqrt{\log(s)/s})$ from the Cramér-Chernoff method.

---

## Learning or Inference Procedure

At inference time, for each input $\mathbf{x}$:
1. Draw sample $\mathbf{w}^{(s)} \sim q(\mathbf{w} \mid \mathcal{D})$ (the variational or MCMC posterior).
2. Evaluate $p(y \mid \mathbf{x}, \mathbf{w}^{(s)})$ and update the running mean $\hat{\mathbf{p}}_s$.
3. Update the confidence sequence bounds $(L_s^c, U_s^c)$.
4. Check the stopping criterion for the chosen inference task.
5. If satisfied, return the prediction with its guarantee; otherwise, draw sample $s+1$.

No training changes are needed; the method wraps any existing BNN.

---

## What the Guarantee Says

The formal guarantee is: for a desired confidence level $1-\alpha$ and any stopping time $\tau$ induced by the confidence sequence, the decision made at time $\tau$ has error probability at most $\alpha$. This holds for any input distribution and any realisation of the Monte Carlo samples, without requiring the posterior approximation to be perfect.

---

## Experimental Findings

- On image classification benchmarks, the method assigns **more samples to ambiguous inputs** (near decision boundaries) and **fewer to easy inputs**, naturally concentrating compute.
- **Overall inference latency** is reduced compared with a fixed-$S$ budget that provides the same average reliability.
- The confidence sequence bounds are tight in practice: the average sample count closely tracks the input difficulty as measured by entropy.
- For the threshold-decision task (uncertainty flagging), the method reduces unnecessary high-sample expenditure on clearly low- or high-uncertainty inputs.

---

## Ablations and Interpretation

**Confidence level $\alpha$:** Lower $\alpha$ (stricter guarantees) requires more samples per input; the relationship is approximately logarithmic in $1/\alpha$.

**Posterior quality:** With a better-calibrated posterior approximation, the CS bounds tighten faster; with an over-dispersed posterior, the method still provides a valid guarantee but requires more samples.

**Comparison with fixed budget $S$:** At matched reliability, the adaptive method uses fewer total samples on typical datasets; the savings are largest on datasets with skewed difficulty distributions (many easy and few hard examples).

**Comparison with heuristic early stopping:** The heuristic gives no guarantee and empirically fails on high-entropy inputs where the running mean is slow to converge.

---

## Reference

Fabian Denoodt, Sibylle Hess.
**When Has a Bayesian Neural Network Sampled Enough? Adaptive Inference Time with Statistical Guarantees.**
NeurIPS 2026 Workshop (E-Values: From Statistics to ML). arXiv:2610.12212, October 2026.
https://arxiv.org/abs/2610.12212
