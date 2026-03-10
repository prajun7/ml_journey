# Choice of K Dimensions — Justification

## Iris Dataset: K = 2

**Justification from explained variance (Figure 1a):**
The PCA explained variance plot shows that the first two principal components capture a cumulative 95.8% of the total variance:

| Component | Individual Variance | Cumulative Variance |
|-----------|-------------------|-------------------|
| PC1       | 73.0%             | 73.0%             |
| PC2       | 22.9%             | **95.8%**         |
| PC3       | 3.7%              | 99.5%             |
| PC4       | 0.5%              | 100.0%            |

K=2 is selected because the cumulative variance exceeds the 95% threshold at two components. PC3 and PC4 together account for only ~4.2% of variance, meaning the dominant data structure is well captured by 2 dimensions. Additionally, for a 3-class problem, LDA can produce at most C−1 = 2 discriminant components. Choosing K=2 ensures the same number of dimensions is used consistently for both PCA and LDA across all analyses.

**Consistency across methods:**
- PCA: 4D → 2 PCs (K=2)
- LDA: 4D → 2 LDs (K=2, which equals min(K, C−1) = min(2, 2) = 2)
- Visualizations in Figures 2a and 3a plot the first 2 of these K=2 dimensions (i.e., all retained dimensions are shown).

---

## Indian Pines Dataset: K = 3

**Justification from explained variance (Figure 1b):**
The PCA explained variance plot for the 202-band Indian Pines dataset shows that the first three principal components cross the 95% cumulative variance threshold:

| Component | Individual Variance | Cumulative Variance |
|-----------|-------------------|-------------------|
| PC1       | 70.1%             | 70.1%             |
| PC2       | 23.5%             | 93.6%             |
| PC3       | 2.1%              | **95.6%**         |
| PC4       | 0.8%              | 96.4%             |
| PC5       | 0.4%              | 96.8%             |
| ...       | <0.4% each        | ...               |

K=3 is selected because it is the smallest number of components where the cumulative explained variance reaches ≥95%. The steep drop-off after PC2 — with each subsequent PC contributing less than ~2% — indicates that 3 components capture nearly all of the dominant spectral variation. The remaining 199 components collectively account for only ~4.4% of variance, primarily representing noise and minor spectral variation.

**Consistency across methods:**
- PCA: 202D → 3 PCs (K=3)
- LDA: 202D → 3 LDs (K=3, which equals min(K, C−1) = min(3, 15) = 3)
- Visualizations in Figures 2b and 3b plot the first 2 of these K=3 dimensions for 2D scatter plots, as required for visualization. The full K=3 dimensions are used in all subsequent classification analyses (Figures 5 and 7).

---

## Summary

| Dataset      | K chosen | Cumulative Variance at K | PCA dims | LDA dims | Same K? |
|-------------|----------|-------------------------|----------|----------|---------|
| Iris         | 2        | 95.8%                   | 2 PCs    | 2 LDs    | Yes     |
| Indian Pines | 3        | 95.6%                   | 3 PCs    | 3 LDs    | Yes     |

The 95% cumulative explained variance threshold is used as the criterion for selecting K in both datasets. This threshold balances dimensionality reduction (removing redundant/noisy dimensions) against information preservation (retaining enough variance to represent the data faithfully). The same K is applied to both PCA and LDA for each dataset, ensuring a fair and consistent comparison across all analyses and classification experiments.
