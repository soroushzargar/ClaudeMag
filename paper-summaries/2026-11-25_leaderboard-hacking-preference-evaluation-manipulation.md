# Leaderboard Hacking: Preference-Based Model Evaluations are Vulnerable to Manipulation

**Authors:** (NeurIPS 2026 oral; full author list at proceedings)
**arXiv:** 2611.14769 (cs.LG, cs.AI)
**Submitted:** November 2026
**Venue:** NeurIPS 2026 (oral)

---

## Headline

Three targeted manipulation strategies—strategic voting, strategic prompting, and strategic nomination—can move a model up 15–21 places each on arena-style leaderboards; combined, they produce rank boosts of up to 45 places, exposing the Bradley–Terry model as structurally gameable.

---

## Key Findings

- **Strategic voting** submits carefully chosen pairwise votes to maximize a target model's rank; each vote may be cast against models other than the target.
- **Strategic prompting** identifies prompts that disproportionately elicit the target model's strong responses; these prompts are preferentially submitted to the arena.
- **Strategic nomination** inserts irrelevant decoy models into the evaluation pool to distort the Bradley–Terry score distribution in the target's favor.
- Each individual strategy boosts rank by up to **15–21 places**; all three applied together boost rank by up to **45 places**.
- The vulnerabilities are **not specific to any one platform**: they apply to any pairwise preference-based evaluation that uses Bradley–Terry scoring, including LMArena.
- Rank manipulation of this magnitude is achievable with resources available to individual actors, not just well-funded organizations.

---

## Motivation and Problem

Arena-style leaderboards such as LMArena have become among the most widely cited benchmarks for comparing large language models. Their appeal is that they aggregate organic human preferences via pairwise battles rather than relying on a fixed test set. The Bradley–Terry model—a probabilistic ranking framework—is typically used to convert pairwise win/loss statistics into a global ranking.

The intuition behind these leaderboards is that organic human preferences aggregate to signal true quality, and that any individual actor's ability to game them is limited. This paper challenges that intuition directly.

The key observation is that the Bradley–Terry model is a statistical estimator trained on whatever data it receives. Its outputs are only as reliable as the integrity of the input data. If participants can selectively influence the data-generating process—which prompts are submitted, how votes are cast, which models compete—then the statistical aggregation may amplify rather than average out strategic behavior.

---

## Method

### Strategic Voting

The attacker uses a "voting oracle" strategy: rather than voting randomly, they target matches between the victim model and models whose loss most benefits the target. Specifically, they upvote the target model in matches against strong competitors (raising the target's perceived score) and downvote it in matches against weak models (avoiding floor effects). The oracle knows the current rating estimates and allocates a fixed vote budget optimally.

### Strategic Prompting

The attacker maintains a prompt portfolio biased toward the target model's strengths. For each category (coding, reasoning, creative writing), they select prompts from a distribution tuned to elicit disproportionately strong responses from the target. When these prompts enter the arena, the target model benefits from a tilted evaluation distribution.

### Strategic Nomination

The attacker introduces decoy models that are selectively weak against specific competitors but strong against the target. Because Bradley–Terry treats the full graph of pairwise outcomes, introducing a model that consistently loses to the target but beats others shifts scores in a way that benefits the target without requiring direct vote manipulation.

### Bradley–Terry Vulnerability Analysis

The paper provides a formal analysis of why Bradley–Terry is susceptible. The key result is that a small perturbation to the win matrix—achievable by any of the three strategies—can produce a rank change that is unbounded in the limit of large model pools, and grows roughly as $O(\log N)$ in the number of models $N$. For pools of the size used in production arenas (50–100 models), the analysis predicts rank boosts consistent with the empirical observations.

---

## Technical Details

The formal model treats the arena as a directed graph where edge weights represent pairwise win probabilities. Bradley–Terry score estimation is equivalent to finding a maximum likelihood solution to the resulting logistic regression problem. The paper shows that gradient manipulation of the win-probability matrix—feasible through each of the three strategies—can steer the solution toward arbitrarily high ranks for a target model.

The analysis extends to any variant of Bradley–Terry, including the Davidson extension (for ties) and weighted variants that down-weight votes from low-quality raters. None of these mitigations eliminate the structural vulnerability.

---

## Experimental Findings

The authors evaluate all three strategies on a simulated LMArena environment with 72 models and 50,000 preference votes. Starting positions range across the leaderboard; the target model begins at median rank.

| Strategy | Median rank gain | Max rank gain |
|---|---|---|
| Strategic voting | +13.2 | +21 |
| Strategic prompting | +10.8 | +18 |
| Strategic nomination | +7.4 | +15 |
| All three combined | +31.6 | +45 |

The rank gains are robust across starting positions and model pool sizes. Even partial execution of the strategies—e.g., 20% of the optimal vote budget—produces rank gains of 5–8 places, well within the range that matters for model visibility and downstream trust.

---

## Significance

This paper is a methodological warning shot for the ML evaluation community. Arena-style leaderboards have displaced fixed benchmarks as the gold standard for LLM quality assessment, but the paper shows the underlying statistical machinery is structurally gameable. The magnitude of the rank distortion—up to 45 places—is not marginal. It is enough to move a model from outside the top 20 to within the top 5.

The implications extend beyond LMArena to any evaluation that aggregates pairwise preferences, including internal A/B testing platforms, research leaderboards, and commercial deployment decisions. The paper does not propose a complete fix, but suggests that adversarial robustness should become a first-class criterion for leaderboard design alongside statistical efficiency.
