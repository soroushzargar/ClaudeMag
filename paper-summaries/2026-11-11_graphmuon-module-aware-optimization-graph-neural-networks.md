# Module-Aware Optimization for Graph Neural Networks (GraphMuon)

**arXiv:** 2611.08379  
**Authors:** Guy Hadad, Haggai Roitman, Moshe Eliasof, Bracha Shapira  
**Venue:** NeurIPS 2026  
**Submitted:** 5 November 2026  
**URL:** https://arxiv.org/abs/2611.08379

---

## Summary

The Muon optimizer improves convergence over AdamW by orthogonalizing gradient matrices using Newton-Schulz iterations, implicitly implementing a matrix-valued second-order update. However, Muon was designed for standard neural network layers where the input covariance is roughly isotropic across training. In GNNs, graph propagation (multiplication by the normalized adjacency matrix A) systematically distorts the covariance of layer inputs, causing Muon's update to amplify rather than normalize step magnitudes. GraphMuon corrects this by adapting Muon's metric to the covariance structure induced by graph-propagated representations, restoring the normalization property that makes Muon effective and achieving 2.13–7.38× wall-clock speedups over AdamW across 15 benchmark configurations.

---

## Method

**Muon's implicit geometry.** For a linear layer with weight matrix W and input X, Muon computes the polar factorization of the gradient G = ∂L/∂W, producing an update U such that ||U||_F = ||G||_F but U has orthonormal rows. This is equivalent to preconditioned gradient descent with preconditioner (G G^T)^{-1/2}, which normalizes by the gradient's row-covariance.

**The GNN complication.** In a GNN layer, the effective input is H = A X, where A is the graph diffusion operator and X is the previous-layer representation. The linear layer then computes H W^T. The gradient with respect to W is G = ∂L/∂W = H^T ∂L/∂Z, where ∂L/∂Z is the upstream gradient. Vanilla Muon orthogonalizes G ignoring H; this normalization is correct only if H has isotropic covariance, which graph propagation violates systematically.

**GraphMuon's correction.** GraphMuon replaces the standard Frobenius inner product used by Muon with a graph-metric inner product: ⟨U, V⟩_Σ = tr(U Σ V^T), where Σ = H^T H / n is the empirical covariance of the propagated input. The polar factorization under this metric gives the update U_Σ = (G Σ^{-1})(G Σ^{-1} G^T)^{-1/2} Σ^{1/2}. GraphMuon approximates Σ from mini-batch statistics and updates the estimate with an exponential moving average, keeping the per-step overhead modest.

**Step-size normalization.** Vanilla Muon normalizes by gradient matrix spectral norm. GraphMuon normalizes by the update's induced effect on the layer representation — specifically by ||U_Σ H^T||_F — so that the effective learning rate measures actual change in layer output rather than raw parameter change.

---

## Experimental Setup

Evaluated on 15 dataset-architecture configurations spanning:
- **Node classification:** Cora, Citeseer, ogbn-arxiv with GCN, GraphSAGE, GAT
- **Inductive graph classification:** TUDatasets with GIN
- **Graph regression:** ZINC with GNN+

Baselines: AdamW, standard Muon. All methods are tuned for peak performance; comparisons use wall-clock time-to-target as the primary metric.

---

## Results

| Metric | GraphMuon vs. AdamW |
|--------|----------------------|
| Time-to-target (wall-clock) | 2.13–7.38× faster (13/15 configs) |
| Predictive accuracy | Competitive or superior in all 15 configs |
| Peak accuracy vs. AdamW | Marginally higher in 9/15 configs |

Vanilla Muon fails to converge on 4/15 configurations due to graph-induced update amplification; GraphMuon converges on all 15.

---

## Key Insight

Graph propagation is a linear operator that stretches input representations along certain directions in proportion to the graph's eigenspectrum. Any optimizer that ignores this stretching will mis-calibrate its update magnitude. GraphMuon is the first optimizer to explicitly account for the covariance structure introduced by message passing, and its strong results confirm that this correction is both necessary (vanilla Muon fails) and sufficient (GraphMuon matches or exceeds AdamW).

---

## Limitations

- The covariance Σ is estimated from mini-batch activations; sparse graphs with high heterophily may produce noisy estimates.
- The theory is developed for single-layer GNNs; multi-hop propagation (deep GNNs) adds complexity not fully addressed.
- Comparison is limited to AdamW and vanilla Muon; results against other adaptive optimizers (e.g., SOAP, Shampoo) are not reported.
