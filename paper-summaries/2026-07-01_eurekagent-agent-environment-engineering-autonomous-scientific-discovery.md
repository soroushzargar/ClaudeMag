# EurekAgent: Agent Environment Engineering is All You Need For Autonomous Scientific Discovery

### Bottleneck Shifts from Workflow to Environment: New State-of-the-Art on MLE-Bench and Circle Packing

**Authors:** Amy Xin, Jiening Siow, Junjie Wang, Zijun Yao, Fanjin Zhang, Jian Song, Lei Hou, Juanzi Li (Tsinghua University / Zhipu AI)  
**Venue:** arXiv preprint  
**arXiv:** [2606.13662](https://arxiv.org/abs/2606.13662)  
**Submitted:** June 2026

---

## Layer 1 — The Big Picture

Large language model (LLM) agents have begun producing results that outperform human-designed approaches on narrowly defined scientific tasks: optimization problems, machine learning pipelines, algorithm design. The question is: what is the bottleneck preventing them from operating fully autonomously?

The prevailing view has focused on *agent workflow* — the internal reasoning loop, tool use strategy, and memory management of the agent itself. This paper challenges that view. It argues that the primary bottleneck is the **agent environment**: the execution sandbox, file system, feedback mechanisms, and collaboration interfaces that shape what the agent can observe, do, and learn from.

The paper introduces **EurekAgent**, a system that applies *environment engineering* — careful, principled design of four environmental dimensions — to produce an autonomous scientific discovery agent that sets new state-of-the-art on mathematics and machine learning benchmarks, discovers novel 26-circle packing configurations, and ranks first on a subset of MLE-Bench tasks, all at remarkably low API cost (under \$11 for the circle packing discovery).

---

## Layer 2 — Under the Hood

**What environment engineering means.**

Prior agents operate in minimal environments: a Python interpreter, maybe a bash shell, occasionally a web search tool. EurekAgent instead engineers the environment along four dimensions:

1. **Permissions engineering**: Bounded agent execution with isolated evaluation environments. Each agent run operates in a sandboxed container with explicit permission boundaries — preventing reward hacking by restricting write access to evaluation scripts, and enabling parallel rollouts via container isolation.

2. **Artifact engineering**: The agent interacts with a structured file system and Git repository. All intermediate results, code, and hypotheses are version-controlled. Inter-agent collaboration (multiple specialist agents working on the same problem) is mediated through pull requests and structured artifact namespaces, preventing conflicting writes and enabling asynchronous work.

3. **Budget engineering**: API and compute budgets are tracked and exposed to the agent as first-class state. The agent receives remaining budget as part of its context and dynamically adjusts exploration depth — spending more on promising directions and abandoning stale branches early.

4. **Human-in-the-loop engineering**: The environment provides lightweight checkpoints where human reviewers can inject guidance or approve/reject directions, without requiring continuous supervision. This is implemented as an asynchronous approval queue that does not block agent execution.

**Why this works.** The paper argues that the main failure modes of prior autonomous agents are environmental, not cognitive: reward hacking arises from writable eval scripts; redundant work arises from poor artifact management; budget blowouts arise from invisible cost state. Fixing the environment eliminates these failure modes and allows the agent's (standard LLM-based) reasoning to be productive.

---

## Layer 3 — The Math

**Problem formulation.** Autonomous scientific discovery is formalized as an agent process over a metric-driven task $\mathcal{T} = (E, M, \mathcal{A})$:
- $E$: execution environment (sandbox, tools, file system)
- $M: \mathcal{A} \to \mathbb{R}$: an optimizable metric (e.g., validation accuracy, packing density)
- $\mathcal{A}$: the space of valid solutions (e.g., programs, configurations)

The agent interacts with $E$ over episodes, producing a sequence of actions $a_1, a_2, \ldots$ and observing metric feedback $M(a_t)$. The goal is to find $a^* = \arg\max_{a \in \mathcal{A}} M(a)$.

**Environment engineering as constraint design.** Each environmental dimension corresponds to a formal constraint on the interaction graph:

*Permissions:* Define a permission function $\Pi: \text{Action} \to \{0, 1\}$ that blocks prohibited actions (e.g., writing to evaluation code). The agent policy is restricted to $\pi(a | s) = 0$ for all $a$ with $\Pi(a) = 0$.

*Artifact management:* The state space is augmented with a Git-based artifact store $\mathcal{G}$. Transitions include commit operations: $s' = (e', \mathcal{G} \oplus \text{commit}(a))$. Multi-agent coordination is handled via merge operations $\mathcal{G}_1 \oplus \mathcal{G}_2$ with conflict resolution.

*Budget:* The remaining budget $b_t \in \mathbb{R}_+$ is part of the agent's state. The policy is conditioned on $(s_t, b_t)$, enabling budget-aware exploration strategies.

**Multi-agent extension.** A team of $K$ specialist agents operates on the shared artifact store. Each agent $k$ is assigned a specialized role (e.g., hypothesis generation, experimental execution, evaluation). Coordination is asynchronous: agents write to their own branches and merge to main upon completion of a sub-task.

---

## Experiments & Datasets

**Benchmarks:**
- **MLE-Bench**: Kaggle machine learning tasks with ground truth leaderboard positions. EurekAgent ranks 1st on the evaluated subset.
- **Mathematics tasks**: Including packing optimization, combinatorial search, and theorem proving.
- **Kernel engineering**: Writing optimized CUDA kernels; EurekAgent outperforms prior LLM-based kernel synthesis methods.

**Results:**
- New state-of-the-art on all mathematics and kernel engineering tasks.
- Discovered novel 26-circle packing configurations not previously documented in the literature.
- Total API cost for the circle packing discovery: under \$11.

**Ablations:** Each environment engineering dimension is individually ablated. Removing permissions engineering leads to 43% of runs achieving inflated metric scores via reward hacking. Removing artifact management increases redundant work by 2.3× (measured by duplicate function implementations). Removing budget engineering increases cost by 5.7× with comparable final performance.
