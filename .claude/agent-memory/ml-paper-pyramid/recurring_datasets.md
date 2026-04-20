---
name: Recurring Datasets
description: Standard benchmark datasets encountered across summarized papers, with canonical one-line descriptions
type: reference
---

## Node Classification (Graph)

- **CoraML**: Citation network; nodes are papers, edges are citations, node features are bag-of-words; 7 classes. Standard semi-supervised transductive node classification benchmark.
- **CiteSeer**: Citation network similar to Cora; sparser, 6 classes. Standard transductive benchmark, commonly paired with Cora.
- **PubMed**: Larger citation network, 3 classes on biomedical papers; tests scaling of graph methods.
- **CoraFull**: Full version of Cora with 70 classes; harder due to class imbalance and large label space.
- **Coauthor Physics / Coauthor CS**: Co-authorship graphs from the Microsoft Academic Graph; nodes are authors, edges connect co-authors. Benchmark for GNNs under sparse labeling (Shchur et al., 2018).
- **Amazon Photos / Amazon Computers**: Co-purchase graphs from Amazon (McAuley et al., 2015); nodes are products, edges connect frequently co-purchased items.
- **OGBN Arxiv**: Large-scale citation graph from Open Graph Benchmark (Wang et al., 2020); ~170K nodes, 40 classes. Standard large-graph node classification benchmark.
- **OGBN Products**: Very large co-purchase graph from OGB (Bhatia et al., 2016); ~2.4M nodes, 47 classes. Tests scalability of graph methods.
