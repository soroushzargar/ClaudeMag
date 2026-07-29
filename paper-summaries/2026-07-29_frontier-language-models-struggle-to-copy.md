# Frontier Language Models Struggle to Copy: Text Can Be Better Viewed in 2D

**Authors:** Haodong Wen, Yiran Zhang, Yingfa Chen, Kaifeng Lyu  
**Affiliation:** Tsinghua University  
**Venue:** arXiv:2607.16072  
**Date:** July 15, 2026  
**URL:** https://arxiv.org/abs/2607.16072

---

## Summary

Frontier language models solve complex mathematical reasoning, write production-quality code, and pass professional exams—yet they consistently fail at a task any human child can perform: copying a string exactly. This paper diagnoses that failure as structural and architectural, proposes a fix in the form of 2D-RoPE, and validates it at scale.

## Problem

Copying an input string $x = (x_1, \ldots, x_N)$ to output $y = (x_1, \ldots, x_N)$ requires retrieving each $x_k$ from a fixed absolute position. With standard 1D RoPE, however, the relative offset between query position (in the output) and key position (in the input) changes as the output length grows. The model must thus learn to copy through local-context pattern matching—a fragile shortcut that breaks on long or repetitive strings.

Experiments on frontier models (GPT-4o, Claude 3.7, Gemini 2.0) confirm the failure: accuracy on exact copy tasks drops below 50% on strings longer than 512 tokens, and falls precipitously on repetitive patterns where local-context shortcuts are ambiguous.

## Method: 2D-RoPE

The paper introduces **2D-RoPE**, which organizes a sequence of $N$ tokens into a 2D grid with $R$ rows and $C$ columns ($N = R \times C$). Each token receives:
- A **row ID** $r_i = \lfloor i/C \rfloor$
- A **column ID** $c_i = i \bmod C$

Rotary positional encoding is applied along each dimension independently. Under this view, the copy task becomes: for each output position $(r, c)$, retrieve the input token at the same column $c$ in row $r$. The relative column offset is fixed at 0 regardless of sequence length—making copying a task of zero-offset column retrieval.

## Technical Formulation

Standard RoPE encodes position $p$ with rotation matrix $R(p\theta_i)$ applied to query and key vectors at frequency $\theta_i$. The attention score depends on the relative position $p - q$:

$$\text{Attn}_{i,j} = \text{Re}\left[q_i^* \cdot k_j \cdot e^{i(p_i - p_j)\theta}\right]$$

In 2D-RoPE, position is split into row and column components:

$$\text{Attn}_{i,j} = \text{Re}\left[q_i^* \cdot k_j \cdot e^{i[(r_i - r_j)\theta_R + (c_i - c_j)\theta_C]}\right]$$

During copying, $c_i = c_j$ (same column in input and output), so the column term vanishes and the attention score depends only on the row offset—a fixed structural relationship that the model can reliably learn.

## Key Results

- **Perfect copying at long range:** Shallow Transformers (6 layers) trained with 2D-RoPE on sequences of length 512 achieve 99.7% exact copy accuracy on length-4096 sequences—8× beyond training range. Standard RoPE achieves 12% on the same sequences.
- **Repetitive patterns:** On sequences with repetition period 4, 2D-RoPE achieves 98.1% accuracy vs. 7.3% for 1D RoPE.
- **Scale confirmation:** 1.4B parameter models trained from scratch with 2D-RoPE on 100B tokens retain the copy advantage while matching standard perplexity on LM benchmarks.
- **Downstream tasks:** 2D-RoPE models show improved performance on information-extraction tasks requiring faithful verbatim retrieval from context.

## Intervention Studies

To confirm the architectural nature of the failure, the authors perform attention-map analysis on frontier models during copy tasks. Models using 1D RoPE attend to local context rather than the exact source position, with attention spread over a ±5 token window rather than spiking at offset 0. 2D-RoPE models show clean offset-0 attention spikes in column-dimension heads.

## Significance

The result has immediate implications for any task requiring verbatim extraction from context: quote extraction, code formatting, structured output generation, and format-following. Systems relying on LLMs to copy or preserve tokens exactly should treat positional encoding as a first-class design decision.
