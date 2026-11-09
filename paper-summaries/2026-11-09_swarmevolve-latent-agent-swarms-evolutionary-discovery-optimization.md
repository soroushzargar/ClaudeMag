# SwarmEvolve: Latent Agent Swarms for Evolutionary Discovery and Optimization

**arXiv:** 2611.08483  
**Authors:** Qile Jiang, George Em Karniadakis (Brown University)  
**Venue:** cs.AI / cs.NE  
**Submitted:** 6 November 2026  
**URL:** https://arxiv.org/abs/2611.08483

---

## Summary

LLM-guided evolutionary search (e.g., OpenEvolve, FunSearch) uses language models to propose and refine programs or mathematical expressions, achieving strong results on hard discovery problems. The bottleneck is throughput: each round requires a full LLM forward pass with lengthy chain-of-thought reasoning per candidate, limiting the number of programs evaluated per unit time. SwarmEvolve addresses this by running agents entirely in latent space, sharing information through KV-cache passing rather than decoding to text. A population of 450 agents (10 swarms × 45 agents) reason in parallel for 40 steps each without ever producing natural language. The result is 5–13× more candidate programs evaluated per hour than state-of-the-art text-based evolutionary frameworks, with equivalent or better solution quality.

---

## Method

**Latent agents.** Each agent is a frozen LLM that processes a fixed context (problem description, partial solution, fitness feedback) and produces a latent representation by running 40 autoregressive steps without applying the unembedding layer. The final hidden state encodes a "proposal" in activation space.

**KV-cache sharing.** Agents in the same swarm share a common prefix KV cache covering the problem description and global best solution. This eliminates redundant computation and allows an agent to implicitly condition on others' reasoning states without text communication.

**Swarm structure.** Ten swarms of 45 agents each run in parallel. Within a swarm, agents have different random seeds but the same prefix cache. After each round:
1. A lightweight decoding head maps the top-k agent hidden states to program tokens (not full trajectories).
2. The decoded programs are executed and scored.
3. The best programs update the global prefix cache for the next round.

**Fitness evaluation.** Programs are executed in a sandboxed environment. Fitness is defined per problem (e.g., algebraic error for symbolic math, forward simulation error for PDE solvers).

---

## Experimental Setup

Evaluated on 21 problems from two domains:
- **Mathematical discovery:** symbolic regression, integer sequence discovery, algebraic identity search.
- **Scientific computing:** PDE solver kernel optimization, numerical integration stencil search.

Baselines: OpenEvolve, ShinkaEvolve, CodeEvolve (all text-based).

---

## Results

- **Throughput:** SwarmEvolve evaluates 5–13× more candidate programs per hour than the best text-based baseline.
- **Time-to-solution:** Reaches the quality of the best competing solution in a median of 3–8 minutes versus 60 minutes for text-based frameworks.
- **Solution quality:** Highest average score across all 21 problems; wins 17 of 21 head-to-head comparisons against the best single baseline.

---

## Key Insight

The critical observation is that the bottleneck in LLM-guided evolutionary search is not reasoning quality but generation throughput. Latent agents bypass the natural-language bottleneck by never decoding intermediate steps, reducing per-candidate cost to a small constant number of transformer layers rather than a full chain-of-thought sequence. KV-cache sharing multiplies this efficiency further by amortizing the common prefix across the entire swarm.

---

## Limitations

- The decoding head (mapping hidden states to programs) is trained separately and may not generalize to program types it was not trained on.
- Agents cannot perform multi-step tool use or external lookups, limiting applicability to self-contained generation tasks.
- The latent reasoning mechanism is not interpretable; it is not clear what the agents "discover" in activation space.
