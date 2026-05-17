# Metrics Documentation

## Physical Deviation Error (PDE)

### Definition

PDE quantifies the $L^2$ divergence between generated trajectory $\mathbf{x}(t)$ and analytical ground truth $\mathbf{x}^*(t)$:

$$\text{PDE} = \frac{1}{N} \sum_{i=1}^{N} \sqrt{\sum_{j=1}^{d} \left(x_j(t_i) - x_j^*(t_i)\right)^2}$$

### Normalization

Per-domain normalization to $[0, 1]$ using maximum observed deviation:

$$\text{PDE}_{\text{norm}} = \frac{\text{PDE}}{\max_{\text{domain}} \|\mathbf{x}^*\|_2}$$

### Interpretation

| PDE Range | Quality |
|-----------|---------|
| 0.00 - 0.05 | Excellent (visually indistinguishable) |
| 0.05 - 0.15 | Good (minor deviations) |
| 0.15 - 0.30 | Fair (noticeable errors) |
| 0.30 - 1.00 | Poor (physically incorrect) |

## Execution Rate

Proportion of generated code snippets that execute without runtime errors:

$$\text{Execution Rate} = \frac{\text{Successful Executions}}{\text{Total Attempts}}$$

## Pass@1

Binary criterion for actionable simulator deployment:

$$\text{Pass@1} = \begin{cases} 1 & \text{if executable AND PDE} < 0.15 \\ 0 & \text{otherwise} \end{cases}$$

## Semantic Metrics

### ROUGE-L

Longest common subsequence F1 score between generated and reference code.

### BERTScore

Token-level similarity using CodeBERT embeddings:

$$\text{BERTScore} = \frac{1}{|x|} \sum_{x_i \in x} \max_{\hat{x}_j} \mathbf{x}_i^\top \hat{\mathbf{x}}_j$$

## Statistical Tests

### Paired t-test

For comparing two models on the same examples:

$$t = \frac{\bar{d}}{s_d / \sqrt{n}}$$

where $d_i = \text{PDE}_{\text{model1}}^{(i)} - \text{PDE}_{\text{model2}}^{(i)}$

### Cohen's d (Effect Size)

$$d = \frac{\bar{x}_1 - \bar{x}_2}{s_{\text{pooled}}}$$

| d | Interpretation |
|---|----------------|
| 0.2 | Small |
| 0.5 | Medium |
| 0.8 | Large |
| 1.2 | Very Large |

### Bonferroni Correction

For $m$ pairwise comparisons:

$$\alpha_{\text{corrected}} = \frac{\alpha}{m}$$
