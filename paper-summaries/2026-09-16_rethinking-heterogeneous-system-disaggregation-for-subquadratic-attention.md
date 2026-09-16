# Rethinking Heterogeneous System Disaggregation for Subquadratic Attention

**arXiv:** 2609.13134  
**Submitted:** September 11, 2026  
**Authors:** Arya Tschand, Yaosheng Fu, Vikram Sharma Mailthody, Nicolai Oswald, Po-An Tsai, Ritchie Zhao, Oreste Villa, Vijay Janapa Reddi, Karu Sankaralingam  
**Affiliation:** Harvard University; NVIDIA; University of Wisconsin-Madison  
**Venue:** arXiv preprint

---

## Headline Finding

SQD (SubQuadratic Disaggregation) introduces fine-grained heterogeneous hardware disaggregation that splits the decode step itself by operator rather than splitting only prefill from decode, exploiting the distinct arithmetic intensity profiles of subquadratic attention variants (sparse, linear, sliding-window) to achieve 31–56% improvement in tokens per joule on frontier LLMs over GPU-only baselines on an 8×B200 heterogeneous system.

---

## Key Findings (Pyramid Layer 2)

1. **Subquadratic attention creates within-decode heterogeneity that existing disaggregation ignores.** Prefill-decode disaggregation routes entire phases to different hardware, but within a single decode step, subquadratic attention layers have sharply different arithmetic intensity from dense FFN layers or standard attention, leaving efficiency on the table.
2. **Sparse attention has two distinct compute phases with different hardware needs.** The top-k selection phase (scanning the full KV cache to find relevant entries) is memory-bound (reads all keys, little computation per read), while the top-k attention plus FFN phase (attending to selected entries and running the feed-forward network) is compute-bound once the selection set is small and static.
3. **Linear and sliding-window attention separate naturally by layer type.** Models like GLM or Gemma mix dense and subquadratic attention layers. Dense attention layers have high arithmetic intensity; subquadratic layers plus FFN are memory-bandwidth-limited at small batch decode. These warrant different hardware.
4. **Energy efficiency, not just throughput, is the primary gain.** By routing memory-bound and compute-bound operators to hardware optimized for each, SQD reduces total energy per token by 31–56%, enabling deployment of frontier models on power-constrained infrastructure.

---

## Methodology (Pyramid Layer 3)

### Background: Subquadratic Attention and Hardware Disaggregation

Full (quadratic) multi-head attention:
$$\mathbf{O} = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V$$
has $O(L^2)$ compute and $O(L)$ KV memory at decode time. For long contexts or large-batch serving, the KV cache becomes the bottleneck. Frontier LLMs now mix subquadratic variants:

- **Sparse attention (e.g., GLM 5.2):** Attend to top-$k$ keys by score; $O(kL)$ attention, but $O(L)$ selection cost.
- **Sliding-window attention (e.g., Gemma 4 31B):** Attend only to the most recent $w$ tokens; $O(wL)$ with constant KV footprint.
- **Linear attention (e.g., Nemotron 3 Ultra):** Replace softmax with a linear kernel $\phi(q)^\top \phi(k)$; $O(L)$ compute via associativity.

**Existing disaggregation** separates prefill (compute-bound, batch size effectively the sequence length) from decode (memory-bound, batch size 1–few). Specialized hardware (NPUs, memory-bandwidth-optimized accelerators) handles the phase it is suited for. But within the decode step itself, different operators have different profiles:

| Operator (decode, batch=1) | Arithmetic Intensity | Bound |
|---|---|---|
| Top-k KV selection (sparse attn) | Low (1 FLOP per 16 bytes) | Memory |
| Top-k attention + FFN | High (once $k \ll L$) | Compute |
| Dense attention layer | Medium | Memory |
| Subquadratic layer + FFN | Low | Memory |

SQD exploits this within-decode heterogeneity.

### SQD: SubQuadratic Disaggregation

**SQD for sparse attention:** Split each decode step into two sub-phases:

1. **Selection sub-phase:** Traverse the full KV cache of length $L$ to compute relevance scores for each key and select top-$k$ entries. This is a dot-product scan: $O(L \cdot d_\text{head})$ reads with $O(L)$ FLOPs — arithmetic intensity $\approx 1$ FLOP/byte. Route to a memory-bandwidth-optimized device (HBM3e-heavy accelerator or CPU with large L3 cache).

2. **Attention + FFN sub-phase:** Attend to the selected $k$ keys and run the FFN. With $k \ll L$, the KV footprint is small and fixed; arithmetic intensity becomes $\gg 1$ FLOP/byte. Route to a compute-optimized GPU (high FP8 FLOP/s).

**SQD for linear/sliding-window attention:** Split by layer type within the decode step:

1. **Dense attention layers:** Attend to the full KV cache (for models with mixed dense + subquadratic). These are memory-bound at decode batch size 1. Route to memory-optimized hardware.

2. **Subquadratic attention layers + FFN:** Linear attention computes $h = \sum_j \phi(k_j)v_j^\top$ incrementally; the state update is compute-bound. Route to compute-optimized hardware.

**Communication cost:** The two sub-phases exchange activation tensors between devices. For typical hidden dimensions ($d = 4096$) and batch sizes (1–8), this is $O(d)$ bytes per token per layer — small compared to the KV data transfer.

---

## Technical Formulation

**Arithmetic intensity** of operator $O$ with $F$ FLOPs and $B$ bytes of memory access:
$$I(O) = \frac{F}{B} \quad \text{(FLOPs/byte)}$$

**Roofline model:** The runtime of operator $O$ on device with compute peak $C$ (FLOPs/s) and bandwidth peak $\beta$ (bytes/s) is:
$$T(O) = \max\!\left(\frac{F}{C},\, \frac{B}{\beta}\right)$$

**SQD routing criterion:** Assign operator $O$ to memory-optimized device if $I(O) < I^*$, otherwise to compute-optimized device. The threshold $I^*$ is the ridge point of the system:
$$I^* = \frac{C}{\beta}$$

On B200 GPU: $C = 4.5 \times 10^{15}$ FP8 FLOPs/s, $\beta = 8 \times 10^{12}$ bytes/s, $I^* \approx 562$ FLOPs/byte.

Top-k selection intensity: $I_{\text{sel}} \approx 1$ FLOPs/byte $\ll I^*$: memory device.
Attention + FFN (with $k \ll L$): $I_{\text{attn}} \approx 10^3$ FLOPs/byte $> I^*$: compute device.

---

## Learning or Inference Procedure

SQD is a system-level optimization applied to existing trained models without any fine-tuning. The disaggregation scheme is determined offline by profiling each layer's arithmetic intensity on a representative workload (batch size 1–8, sequence lengths 8k–128k). The routing table is compiled once and applied at serving time. No changes to model weights or architecture are required.

---

## What the Guarantee Says

The roofline model provides an upper bound: no single device can exceed $\max(F/C, B/\beta)$ per operator. SQD routes each operator to the device where this bound is tightest, guaranteeing that each device operates closer to its own bound than in a GPU-only setup. The energy improvement bound follows from the ratio of energy-per-operation on the two device types: memory-optimized devices consume significantly less energy per byte accessed (e.g., LPDDR or CXL memory) than high-bandwidth GPU HBM.

---

## Experimental Findings

### System: 8×B200 Heterogeneous (4× B200 GPU + 4× memory-bandwidth-optimized accelerators)

### Models and Energy Efficiency

| Model | Attn Type | SQD Tokens/J Gain |
|---|---|---|
| GLM 5.2 | Sparse | +53% |
| Nemotron 3 Ultra | Linear | +31% |
| Gemma 4 31B | Sliding-window | +56% |

Baseline: GPU-only serving (all 8 devices are B200 GPUs).

### Throughput (Tokens/s, Batch=8, Seq=32k)

| Model | GPU-only | SQD | Speedup |
|---|---|---|---|
| GLM 5.2 | 1,840 | 2,710 | 1.47× |
| Gemma 4 31B | 2,190 | 3,080 | 1.41× |

---

## Ablations and Interpretation

- **Selection-only disaggregation (SQD-S):** Routing only the top-k selection phase to memory device accounts for 68% of the total energy gain for sparse attention models.
- **Without layer-type disaggregation (SQD-L):** Routing by phase only (prefill/decode) but not by within-decode layer type reduces energy gain from 53% to 29% for GLM 5.2.
- **Communication overhead:** Cross-device activation transfer adds 7–12% latency but is offset by the 40–55% reduction in time-per-token on the memory-bound operators.
- **Dense models (no subquadratic attention):** SQD provides no benefit — standard prefill/decode disaggregation is already optimal when all attention is dense.

---

## Reference

Arya Tschand, Yaosheng Fu, Vikram Sharma Mailthody, Nicolai Oswald, Po-An Tsai, Ritchie Zhao, Oreste Villa, Vijay Janapa Reddi, and Karu Sankaralingam. **Rethinking Heterogeneous System Disaggregation for Subquadratic Attention.** arXiv:2609.13134, September 2026. https://arxiv.org/abs/2609.13134
