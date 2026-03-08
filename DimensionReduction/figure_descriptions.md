# CS488/588 Homework 3 — Figure and Table Descriptions

## Question 1a: Dimensionality Reduction Visualization

### Figure 1a. PCA Explained Variance — Iris Dataset

Bar chart of the individual explained variance ratio for each of the four principal components of the Iris dataset, overlaid with a cumulative variance line and a 95% threshold marker. PC1 dominates (~73.0%), and the first two PCs together capture approximately 95.8% of the total variance, justifying the choice of K=2 for subsequent analysis.

### Figure 1b. PCA Explained Variance — Indian Pines (First 30 PCs)

Bar chart showing individual explained variance for the first 30 principal components of the Indian Pines hyperspectral dataset (202 bands total), with a cumulative variance line. PC1 alone accounts for ~70% of variance. Three PCs cross the 95% cumulative threshold (95.63%), so K=3 is selected. The steep drop-off after PC1 indicates high spectral redundancy typical of hyperspectral imagery.

### Figure 2a. PCA 2D Visualization — Iris Dataset

Scatter plot of the Iris samples projected onto the first two principal components (PC1 vs. PC2), color-coded by species (setosa, versicolor, virginica). PC1 and PC2 together explain ~95.8% of variance. Setosa forms a clearly separable cluster, while versicolor and virginica overlap slightly, indicating that two PCs are sufficient to reveal the main class structure.

### Figure 2b. PCA 2D Visualization — Indian Pines Dataset

Scatter plot of Indian Pines pixel samples projected onto the first two principal components, color-coded by the 16 land-cover classes. PC1 and PC2 capture the majority of the spectral variance. Significant class overlap is visible in 2D, reflecting the challenge of distinguishing spectrally similar land-cover types (e.g., different crop types) with unsupervised projections alone.

### Figure 3a. LDA 2D Visualization — Iris Dataset

Scatter plot of Iris samples projected onto the first two linear discriminants (LD1 vs. LD2), color-coded by species. Unlike PCA, LDA maximizes between-class separation relative to within-class scatter. Setosa is fully separated on LD1 alone. Versicolor and virginica show improved separation compared to PCA, confirming that LDA is more effective for class discrimination on the Iris dataset.

### Figure 3b. LDA 2D Visualization — Indian Pines Dataset

Scatter plot of Indian Pines samples projected onto the first two linear discriminants, color-coded by the 16 classes. LDA leverages label information to find projections that maximize class separability. Compared to PCA (Figure 2b), some classes form tighter and more distinguishable clusters, though substantial overlap remains among spectrally similar classes due to the high number of classes relative to only two projection dimensions.

---

## Question 2a: Supervised Classification Results

### Figure 4. Classification Accuracy with PCA (K=2) — Iris Dataset

Line plot of overall classification accuracy (%) versus training size (10%–50%) for three classifiers (Naive Bayes, SVM Linear, SVM RBF) applied to PCA-reduced Iris data (2 PCs). All classifiers achieve high accuracy (generally >90%) even at small training sizes, with SVM RBF and SVM Linear typically performing slightly better than Naive Bayes. Accuracy plateaus around 95–98% as training size increases.

### Figure 5. Classification Accuracy with PCA (K=3) — Indian Pines Dataset

Line plot of overall classification accuracy versus training size for the three classifiers applied to PCA-reduced Indian Pines data (3 PCs). SVM RBF generally outperforms the other classifiers. Accuracy improves with more training data, but overall accuracy levels are lower than Iris due to the higher number of classes (16) and greater spectral overlap. Reducing 202 bands to only 3 PCs causes some information loss, which limits classifier performance.

### Figure 6. Classification Accuracy with LDA (K=2) — Iris Dataset

Line plot of overall classification accuracy versus training size for the three classifiers applied to LDA-reduced Iris data (2 LDs). Performance is comparable to or slightly better than PCA-based classification (Figure 4), since LDA projections are optimized for class discrimination. All classifiers reach >95% accuracy with 30%+ training data.

### Figure 7. Classification Accuracy with LDA (K=3) — Indian Pines Dataset

Line plot of overall classification accuracy versus training size for the three classifiers applied to LDA-reduced Indian Pines data (3 LDs). LDA-based features generally yield higher classification accuracy than PCA (Figure 5) for the same dimensionality, since LDA explicitly maximizes class separability. SVM classifiers tend to benefit the most from LDA projections on this dataset.

### Figure 8. Classification Accuracy without Dimensionality Reduction — Iris Dataset

Line plot of overall classification accuracy versus training size for the three classifiers applied to the full (standardized) 4-feature Iris dataset. Performance is very similar to the PCA and LDA cases (Figures 4 and 6), which is expected since Iris has only 4 features and dimensionality reduction removes minimal information. This serves as a baseline for comparison.

### Figure 9. Classification Accuracy without Dimensionality Reduction — Indian Pines Dataset

Line plot of overall classification accuracy versus training size for the three classifiers applied to the full (standardized) 202-band Indian Pines dataset. SVM RBF typically achieves the highest accuracy. Compared to PCA-reduced results (Figure 5), using all 202 bands may yield higher accuracy because no spectral information is discarded, but at the cost of significantly higher computational time. Compared to LDA (Figure 7), the relative performance depends on the classifier and training size.

---

## Table

### Table 1. Class-wise Classification Accuracy (Recall) — Indian Pines, PCA (K=3), 30% Training Size

Table listing the per-class recall (sensitivity) for each of the 16 Indian Pines land-cover classes under all three classifiers (Naive Bayes, SVM Linear, SVM RBF), using PCA-reduced data at 30% training size. An overall (macro-average) recall row is included at the bottom. This table reveals which classes are well-classified versus poorly classified, highlighting that minority classes with few training samples (e.g., Oats, Grass-pasture-mowed) often have lower recall. SVM RBF generally achieves the highest per-class recall across most classes.
