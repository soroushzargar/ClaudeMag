# Computational and Statistical Guarantees of the c-Rectified Flow

**arXiv:** 2608.02487  
**Authors:** Leda Wang, Zhehao Xu, Qiang Liu, Harrison H. Zhou  
**Affiliation:** UT Austin; Yale University  
**Date:** August 2026  
**URL:** https://arxiv.org/abs/2608.02487

---

## Summary

Rectified Flow and flow matching have become the standard training frameworks for modern diffusion and flow-based generative models (Stable Diffusion 3, FLUX, LLaMA video), yet their theoretical basis for learning the optimal-transport (OT) solution has remained unclear. Wang et al. introduce and analyze **c-Rectified Flow**, a cost-aware variant that modifies the flow matching objective so that the learned vector field provably minimizes a user-specified convex transport cost c. The paper provides both computational guarantees (iterative c-rectification monotonically reduces transport cost) and statistical guarantees (sample complexity bounds from finite data), establishing the first complete theoretical framework for cost-aware flow generative modeling.

## Background: Rectified Flow and the OT Gap

Standard rectified flow trains a velocity field v(x, t) to minimize the expected squared distance between a straight-line trajectory from noise to data and the learned trajectory. This is equivalent to learning a coupling between the noise distribution and the data distribution. While elegant and fast to train, standard rectified flow does not guarantee that the learned coupling minimizes any particular transport cost—it may find a valid generative model that is highly suboptimal in terms of the cost structure the user actually cares about (e.g., minimizing L1 displacement, Wasserstein-2 distance, or problem-specific costs in molecule generation or image editing).

## c-Rectified Flow: The Key Modification

For a user-specified convex cost function c(x, y), c-Rectified Flow restricts the optimization domain to vector fields of the form:

$$v_t(x) = \nabla \bar{c}(\nabla f_t(x))$$

where f_t(·) is an arbitrary time-dependent scalar function and c̄ is the convex conjugate (Legendre transform) of c. This constraint ensures the flow transports mass along paths that are locally optimal with respect to c.

The training objective becomes a modified regression problem that remains unconstrained in terms of f_t—the gradient structure is enforced by parameterizing v_t through f_t rather than directly. This is practically important: the neural network learns f_t (an unconstrained scalar function), and v_t is computed as its gradient, making the constraint automatic rather than requiring projected gradient descent or Lagrange multipliers.

## Guarantees Provided

**Computational guarantee:** Iterative application of c-rectification (train, couple, retrain) monotonically reduces the expected c-transport cost while automatically preserving the marginal distributions of noise and data. No margin constraints need to be enforced explicitly—the gradient structure of the parameterization handles it.

**Statistical guarantee:** The paper establishes sample complexity bounds: with n samples from source and target distributions, the c-transport cost achieved by the learned flow converges to the population optimum at a rate that depends on the smoothness of the optimal transport map, the dimensionality of the problem, and the complexity of the neural network class used to parameterize f_t.

**Conditions for OT recovery:** The paper characterizes when c-Rectified Flow recovers the true OT solution (global optimum of the c-transport problem) and when it only guarantees local descent. Sufficient conditions include connected support of the distributions and smoothness of the OT map. Counterexamples show that without these conditions, the gradient constraint does not guarantee global optimality.

## Why It Matters

The theoretical gap between "flow matching works well in practice" and "flow matching provably solves the OT problem" has been a persistent concern for applications where the transport cost is semantically meaningful—e.g., in molecule generation where the cost measures conformational change, or in image editing where it measures perceptual distortion. c-Rectified Flow closes this gap by making the cost function a first-class citizen of the training objective while preserving the practical benefits of flow matching (simple regression objective, no score function estimation, fast inference).

## Key Contributions

1. c-Rectified Flow: parameterization of vector fields through convex conjugate structure to enforce cost-awareness
2. Computational guarantee: monotone c-transport cost reduction under iterative rectification
3. Statistical guarantee: sample complexity bounds for the learned c-flow from finite data
4. Characterization of sufficient conditions for OT recovery and counterexamples showing necessity
