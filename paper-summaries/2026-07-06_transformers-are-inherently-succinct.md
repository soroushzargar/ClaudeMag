# Transformers are Inherently Succinct

### Fixed-Precision Transformers Compress Exponentially More Than RNNs and Temporal Logic

**Authors:** Pascal Bergsträßer, Ryan Cotterell, Anthony W. Lin (ETH Zürich / TU Kaiserslautern)
**Venue:** ICLR 2026 Outstanding Paper Award
**arXiv:** [2510.19315](https://arxiv.org/abs/2510.19315)
**Submitted:** October 2025

---

## Layer 1 — The Big Picture

**What problem does the paper solve, and how does it solve it?**

A long-standing theoretical question asks: how expressive are transformers, and how do they compare to classical formalisms? Prior work has established *equivalences* in what transformers can *recognize* — but equivalence in power does not imply equivalence in *succinctness*. A polynomial-size transformer might describe a language whose smallest equivalent finite automaton is exponentially large. This compression gap is the central question of the paper.

The authors establish tight succinctness bounds for fixed-precision transformers relative to linear temporal logic (LTL), recurrent neural networks (RNNs), and finite automata. The result: transformers are **exponentially more succinct** than both LTL and RNNs, and **doubly exponentially more succinct** than finite automata. As a complexity-theoretic corollary, verifying simple transformer properties is EXPSPACE-complete.

---

## Layer 2 — Under the Hood

**Intuition.**

Succinctness asks: given a language $L$, what is the smallest description of $L$ in each formalism? The paper constructs a sequence of languages $\{L_n\}$ that:
- Admit polynomial-size transformer descriptions
- Require exponentially large LTL formulas and RNNs
- Require doubly-exponentially large finite automata

The key construction exploits transformers' ability to use **positional encodings** and **global self-attention** to encode complex ordering constraints compactly — constraints that LTL must unroll through time and finite automata must track via exponentially many states.

The result is bidirectional: the upper bound shows any fixed-precision transformer can be translated into an LTL formula with at most *exponential* blowup (improving a prior *doubly-exponential* translation), while the lower bound witnesses show the exponential gap is tight.

---

## Layer 3 — The Math

**Main theorem (informal):** There exist families of languages $\{L_n\}_{n \geq 1}$ such that:
- The smallest transformer for $L_n$ has size $\text{poly}(n)$
- The smallest LTL formula for $L_n$ has size $2^{\Omega(n)}$
- The smallest RNN for $L_n$ has size $2^{\Omega(n)}$
- The smallest DFA for $L_n$ has size $2^{2^{\Omega(n)}}$

**Complexity result:** The model-checking problem for fixed-precision transformers — given a transformer $M$ and a property $\varphi$ (in LTL), does $M$ satisfy $\varphi$? — is EXPSPACE-complete.

**Upper bound (translation theorem):** Any fixed-precision transformer with $L$ layers, $H$ heads, and dimension $d$ can be expressed as an LTL formula of size at most $\exp(O(L \cdot H \cdot d))$, a single-exponential bound in model size.

The proof technique uses circuit complexity arguments and the theory of automata on infinite words to relate transformer computation to temporal-logic evaluation.

---

## Experiments & Datasets

This is a theoretical paper. No experiments are conducted; the contributions are fully formal proofs.

**Implications for practice:** The EXPSPACE-completeness of transformer verification means that tools for verifying safety or correctness properties of transformers face fundamental computational barriers — polynomial-time verification is provably impossible unless PSPACE = EXPSPACE.

**Connection to interpretability:** The succinctness result explains, in part, why transformers generalize well with relatively few parameters — they can represent complex dependencies compactly that would require far larger classical models.
