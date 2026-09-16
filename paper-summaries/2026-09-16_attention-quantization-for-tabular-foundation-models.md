# Attention Quantization for Tabular Foundation Models

**arXiv:** 2609.13031  
**Submitted:** September 12, 2026  
**Authors:** Jonas M. Kübler, Benjamin Jäger, Klemens Flöge, Noah Hollmann, Frank Hutter  
**Affiliation:** University of Freiburg; ELLIS Institute Tübingen  
**Venue:** arXiv preprint

---

## Headline Finding

FP8 attention quantization for tabular foundation models (TabPFN, TabICL) achieves 1.7× inference speedup over 16-bit baselines with no accuracy loss on TabArena and BeyondArena benchmarks, by introducing a critical alignment procedure that equalizes quantization error between test and training rows — a unique challenge absent from LLM quantization because tabular FM KV matrices scale quadratically with training set size rather than with context length.

---

## Key Findings (Pyramid Layer 2)

1. **Tabular FM KV matrices are uniquely large and scale differently from LLMs.** In LLMs, KV cache size is proportional to context length (typically thousands of tokens). In tabular FMs such as TabPFN, the KV matrix encodes the entire training dataset: a table with $n$ rows and $d$ features has $O(n^2 \cdot d)$ attention entries per head, and $n$ can reach hundreds of thousands of rows in practice.
2. **Quantization error misalignment breaks accuracy.** Test rows in tabular FM attention attend to all training rows. If the quantization error on test rows (keys/queries) differs systematically from the error on training rows (keys/values), the attention scores are biased, producing incorrect predictions even when overall quantization error is small.
3. **Dynamic per-tensor scale calibration with row alignment solves the problem.** FP8 e4m3fn quantization with scales derived from the tensor itself (not from calibration data) plus a procedure that matches quantization error statistics between test and training row subsets eliminates accuracy degradation.
4. **A custom Triton kernel realizes the theoretical speedup.** Existing FP8 kernels are designed for LLM attention patterns; the tabular FM attention has different block shapes and row groupings. A specialized kernel achieves 1.7× speedup over FP16 baselines with negligible kernel overhead.

---

## Methodology (Pyramid Layer 3)

### Background: Tabular Foundation Models and In-Context Learning

Tabular foundation models learn to classify or regress on new datasets at test time via in-context learning (ICL), encoding the entire training set as a sequence of tokens and predicting test rows in a single forward pass. The attention mechanism for row $i$ (test or training) attends over all $n$ rows:
$$\mathbf{A}_{ij} = \text{softmax}\!\left(\frac{\mathbf{q}_i \mathbf{k}_j^\top}{\sqrt{d}}\right)$$

For a table with $n = 10{,}000$ training rows, the full attention matrix has $n^2 = 10^8$ entries per head per layer. With 16-bit precision and 4 heads, this is:
$$\text{memory} = 4 \times 10^8 \times 2 \text{ bytes} = 800\,\text{MB per layer}$$

Quantizing to 8-bit halves this and doubles effective bandwidth utilization.

### The Quantization Error Misalignment Problem

Standard FP8 quantization assigns a scale $s$ to a tensor $X$ based on its maximum absolute value:
$$s = \frac{\max|X|}{448}, \quad \hat{X} = \text{round}(X / s) \cdot s$$
(448 is the maximum representable value in e4m3fn.)

In a tabular FM, the query matrix $Q$ has two structurally different row groups:
- **Training rows** $Q_\text{train} \in \mathbb{R}^{n \times d}$: encoded from observed $(x_i, y_i)$ pairs.
- **Test rows** $Q_\text{test} \in \mathbb{R}^{m \times d}$: encoded from $x_j$ without labels.

These groups have different feature distributions: training rows include label information in their embeddings; test rows do not. Applying a single scale $s$ to the concatenated $[Q_\text{train}; Q_\text{test}]$ biases the scale toward whichever group has larger magnitude. The group with smaller magnitude is quantized with unnecessarily coarse steps, introducing systematic error in its attention scores.

**Concretely:** if $\max|Q_\text{train}| \gg \max|Q_\text{test}|$, then $s$ is calibrated to training rows and test rows use only the low end of the FP8 range — far fewer distinct values — resulting in test attention logits with high quantization noise.

### Alignment Procedure

OmniKVQuant uses separate, dynamic scales per row group:
$$s_\text{train} = \frac{\max_{i \in \text{train}}|Q_i|}{448}, \quad s_\text{test} = \frac{\max_{j \in \text{test}}|Q_j|}{448}$$

The key: compute $s$ from the rows themselves at forward pass time (dynamic calibration), not from a separate calibration dataset. This ensures the FP8 range is fully utilized for each group, equalizing the relative quantization error:
$$\frac{\delta Q_\text{train}}{s_\text{train}} \approx \frac{\delta Q_\text{test}}{s_\text{test}}$$

where $\delta Q$ denotes the quantization rounding error. Alignment ensures neither group is disproportionately penalized.

### Custom Triton Kernel

The standard FP8 GEMM kernels in cuBLAS and FlashAttention assume square or near-square tile shapes optimized for transformer sequence attention. Tabular FM attention has asymmetric shapes ($m \times n$ where $m \ll n$ for few test rows) and requires separate scale factors per row group within the same kernel call. The paper implements a Triton kernel with:
- Asymmetric block tiling ($m$-side tiles sized to test row count, $n$-side tiles sized to GPU cache lines).
- Per-group scale fusion into the attention score accumulation.
- Direct FP8 accumulation without intermediate FP32 cast on the inner loop.

---

## Technical Formulation

**FP8 e4m3fn format:** 4 exponent bits, 3 mantissa bits, 1 sign bit. Representable range: $[-448, 448]$. Precision per unit of scale: $2^{-3} = 0.125$.

**Per-group dynamic quantization:**
$$\hat{Q}_g = \text{round}\!\left(\frac{Q_g}{s_g}\right) \cdot s_g, \quad s_g = \frac{\|Q_g\|_\infty}{448}$$
for group $g \in \{\text{train}, \text{test}\}$.

**Quantized attention computation:**
$$\mathbf{A}_{ij} = \text{softmax}\!\left(\frac{\hat{\mathbf{q}}_i \hat{\mathbf{k}}_j^\top}{\sqrt{d}}\right), \quad \mathbf{O}_i = \sum_j \mathbf{A}_{ij} \hat{\mathbf{v}}_j$$

where Q, K, V are all quantized to FP8 with group-aligned scales.

**Quantization error bound:** The rounding error per element is bounded by $\epsilon_{q} \leq s_g / 2$. With alignment, the relative error satisfies:
$$\frac{\epsilon_q}{\|Q_g\|_\infty} \leq \frac{1}{2 \times 448} \approx 0.001$$
for both row groups equally.

---

## Learning or Inference Procedure

The quantization is applied entirely at inference time with no retraining or calibration pass on held-out data. The FP8 scales are computed dynamically from the current input tensor at each forward pass. The alignment procedure adds one pass over the input to compute per-group max values before the attention kernel, contributing negligible overhead ($<$1% of total attention compute time).

---

## What the Guarantee Says

The theoretical claim is that per-group dynamic scale alignment ensures equal relative quantization error across test and training rows. Empirically, this translates to no statistically significant accuracy degradation on TabArena (47 classification datasets) and BeyondArena (21 regression datasets) using tabular FMs TabPFN-v3 and TabICLv2. The speedup is guaranteed by the custom Triton kernel on an H100 GPU for table sizes $n \geq 1{,}000$ rows where the attention computation dominates.

---

## Experimental Findings

### Models
- TabPFN-v3 (classification)
- TabICLv2 (classification and regression)

### Benchmarks
- **TabArena:** 47 tabular classification datasets; metric: normalized accuracy (relative to best method per dataset).
- **BeyondArena:** 21 tabular regression datasets; metric: normalized RMSE.

### Main Results

| Model | Precision | Speedup | Acc. Change |
|---|---|---|---|
| TabPFN-v3 | FP16 | 1.00× | — |
| TabPFN-v3 | FP8 (no alignment) | 1.65× | −3.1% |
| **TabPFN-v3** | **FP8 (aligned)** | **1.70×** | **−0.1%** |
| TabICLv2 | FP16 | 1.00× | — |
| TabICLv2 | FP8 (no alignment) | 1.62× | −4.7% |
| **TabICLv2** | **FP8 (aligned)** | **1.68×** | **−0.2%** |

Alignment is critical: without it, FP8 causes 3–5% accuracy degradation; with it, degradation is within noise.

---

## Ablations and Interpretation

- **Without per-group scale (single global scale):** Accuracy drops by 3.1–4.7 points, confirming that the test/training distribution mismatch is the dominant failure mode, not FP8 precision itself.
- **E4m3fn vs. E5m2 format:** E4m3fn (more mantissa bits) performs better on tabular attention than E5m2 (more exponent bits), because the distribution of attention logits is well-concentrated and benefits more from mantissa precision than dynamic range.
- **Kernel tile shapes:** Custom asymmetric tiling provides 18% additional speedup over reusing a standard square-tile FP8 kernel, demonstrating that the tabular attention shape warrants specialized implementation.
- **Scaling with $n$:** Speedup grows with training set size; at $n = 500$ the improvement is 1.3×; at $n = 10{,}000$ it reaches 1.9×, reflecting that larger attention matrices are more bandwidth-bound and benefit more from quantization.

---

## Reference

Jonas M. Kübler, Benjamin Jäger, Klemens Flöge, Noah Hollmann, and Frank Hutter. **Attention Quantization for Tabular Foundation Models.** arXiv:2609.13031, September 2026. https://arxiv.org/abs/2609.13031
