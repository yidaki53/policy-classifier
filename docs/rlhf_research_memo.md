---
_agent_frontmatter:
  id: "docs/rlhf_research_memo"
  purpose: "Repository markdown document."
  steward: "repo"
  edit_policy: "manual"
---

# Reinforcement Learning from Human Feedback (RLHF) / Deep Learning Meta-Learner

## Research Memo and Findings

### Context

The swedish_parliament_policy_classifier project uses a multi-signal ensemble
(keywords, embeddings, zero-shot NLI, topic distributions) with a LightGBM
meta-classifier. The goal is to classify parliamentary motions by ideology
(7-class spectrum) purely from policy content, without party-bias leakage.

### Why RLHF Is Not Recommended Here

1. **No Human Rater Pool**: PPO/DPO requires preference pairs (A > B) from
   human annotators judging which of two model outputs is better. The project
   has ~2,700 gold labels but no pairwise preference dataset and no budget for
   human RLHF labeling.

2. **Classification, Not Generation**: RLHF shines in generative models
   (chatbots, summarization) where the reward model evaluates open-ended text.
   A 7-class classifier outputs a probability distribution over fixed labels;
   there is no generative trajectory to optimize with policy gradients.

3. **Reward Signal Is Already Handled**: The LightGBM ensemble already uses
   balanced class weights and an LLM-as-judge fallback for low-confidence
   predictions. Any RLHF reward model would need to replicate the same
   classification correctness logic, which the classifier already captures.

4. **Party-Bias Penalty Can Be Simpler**: Instead of RLHF, a direct penalty
   can be added during training: if a classifier's accuracy on cross-party
   motions is lower than within-party, add an auxiliary loss term. This is
   straightforward supervised learning, not reinforcement learning.

### What Was Implemented Instead

To explore deep learning meta-learners as an alternative to LightGBM, two
approaches were implemented and compared:

| Approach | Test Accuracy | Status |
|---|---|---|
| LightGBM ensemble (baseline) | **71.8%** | Current best |
| Direct transformer (KBLab BERT, 512 tokens) | 69.7% | Slightly below baseline |
| Small MLP on feature vectors | TBD | Implemented, running |

The transformer classifier is end-to-end (text -> 7 classes) but underperforms
the feature-engineered ensemble, suggesting the handcrafted signals (keywords,
topics, embeddings) capture useful inductive biases that the transformer misses
with limited data (~1,350 train samples).

### Hybrid Ensemble Recommendation

The most promising path is a **hybrid ensemble**:
- **Transformer features**: Use the fine-tuned BERT's [CLS] embedding as an
  additional feature in the LightGBM meta-classifier.
- **Confidence gating**: For high-confidence transformer predictions (>0.85),
  use them directly; for low-confidence, fall back to the full ensemble.

This leverages the transformer's ability to capture unobserved linguistic
patterns while retaining the ensemble's robust keyword/topic signals.

### MLP Meta-Learner

A small PyTorch MLP (2 hidden layers, BatchNorm, Dropout) was implemented
in `scripts/train_mlp_meta_learner.py` to test whether a neural meta-learner
outperforms LightGBM on the same feature vectors. If it matches or exceeds
71.8%, it becomes a viable replacement with easier GPU batching and
integration into an RL-like online update loop.

### RLHF Future Work (If Budget Allows)

If the project later obtains a large pairwise preference dataset (e.g.,
comparing two classifier predictions on the same motion, with human experts
choosing the better one), PPO or DPO could be explored to:
- Reward correct rare-class predictions (far-left, far-right).
- Penalize overconfident center predictions (the majority class bias).

Until then, supervised methods with class balancing and active learning
remain the highest-ROI path.
