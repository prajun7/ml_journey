# CS488/588 Homework 3 — Figure and Table Descriptions

## Question 1a: Dimensionality Reduction Visualization

### Figure 1a. PCA Explained Variance — Iris Dataset

Bar chart of the individual explained variance ratio for each of the four principal components of the Iris dataset, overlaid with a cumulative variance curve (red) and a 95% threshold line (green dashed). PC1 accounts for ~73.0% and PC2 for ~22.9% of the total variance. The cumulative line crosses 95% at PC2, reaching approximately 95.8%, which justifies selecting K=2 for subsequent analysis. PC3 and PC4 contribute negligibly (~3.7% and ~0.5%, respectively).

### Figure 1b. PCA Explained Variance — Indian Pines (First 30 PCs)

Bar chart of individual explained variance (orange bars) for the first 30 of 202 principal components of the Indian Pines hyperspectral dataset, with a cumulative variance curve (red) and a 95% threshold line (green dashed). PC1 alone captures ~70.1% of variance and PC2 adds ~23.5%. The cumulative line crosses the 95% threshold at PC3 (95.63%), so K=3 is selected. The steep drop-off after PC2 — with remaining PCs each contributing less than ~1% — reflects the high spectral redundancy typical of hyperspectral imagery.

### Figure 2a. PCA 2D Visualization — Iris Dataset

Scatter plot of the 150 Iris samples projected onto PC1 (73.0% variance) vs. PC2 (22.9% variance), color-coded by species: setosa (red), versicolor (green), and virginica (orange). Setosa forms a clearly isolated cluster in the upper-left region of the plot, while versicolor and virginica overlap slightly along the PC1 axis, indicating that two PCs effectively reveal the primary class structure of the Iris dataset.

### Figure 2b. PCA 2D Visualization — Indian Pines Dataset

Scatter plot of 10,249 Indian Pines pixel samples projected onto PC1 (70.1% variance) vs. PC2 (23.5% variance), color-coded by 16 land-cover classes. Despite capturing ~93.6% of the total variance in two dimensions, significant class overlap is visible throughout the plot. Several classes (e.g., different crop types) cluster together, reflecting the difficulty of separating spectrally similar land-cover types with an unsupervised projection like PCA.

### Figure 3a. LDA 2D Visualization — Iris Dataset

Scatter plot of the Iris samples projected onto LD1 vs. LD2, color-coded by species. Unlike PCA, LDA maximizes between-class separation relative to within-class scatter. All three species are well separated: setosa (red) is fully isolated on the right, versicolor (green) occupies the center, and virginica (orange) clusters on the left. The class separation is markedly improved over PCA (Figure 2a), confirming that LDA is more effective for supervised class discrimination on this dataset.

### Figure 3b. LDA 2D Visualization — Indian Pines Dataset

Scatter plot of Indian Pines samples projected onto LD1 vs. LD2, color-coded by 16 classes. Compared to PCA (Figure 2b), LDA leverages label information to produce projections with tighter, more distinguishable clusters for several classes. Some classes (e.g., Class 8, Class 6, Class 14) form more compact groups. However, substantial overlap persists among spectrally similar classes, which is expected given that 16 classes are projected into only two discriminant dimensions.

---

## Question 2a: Supervised Classification Results

### Figure 4. Classification Accuracy with PCA (K=2) — Iris Dataset

Line plot of overall classification accuracy (%) versus training size (10%–50%) for three classifiers — Naive Bayes (blue), SVM Linear (red), SVM RBF (green) — applied to PCA-reduced Iris data (2 PCs). All three classifiers perform closely together in the ~88–93% range. SVM Linear achieves slightly higher accuracy than the others. Performance is relatively stable across training sizes, indicating that 2 PCs are sufficient for reasonable classification, though some discriminative information is lost compared to using all 4 features.

### Figure 5. Classification Accuracy with PCA (K=3) — Indian Pines Dataset

Line plot of overall classification accuracy versus training size for the three classifiers applied to PCA-reduced Indian Pines data (3 PCs). Naive Bayes and SVM Linear remain nearly flat at ~55–56% across all training sizes. SVM RBF stands out, improving from ~57% at 10% training to ~65% at 50% training. The relatively low accuracies reflect severe information loss when compressing 202 spectral bands into only 3 principal components — while these capture 95% of variance, they discard subtle spectral differences needed to discriminate 16 land-cover classes.

### Figure 6. Classification Accuracy with LDA (K=2) — Iris Dataset

Line plot of overall classification accuracy versus training size for the three classifiers applied to LDA-reduced Iris data (2 LDs). All three classifiers achieve ~96–98% accuracy, which is significantly higher than PCA-based classification (Figure 4, ~88–93%). Naive Bayes and SVM RBF perform nearly identically at ~97–98%, while SVM Linear is marginally lower at ~96%. The marked improvement over PCA demonstrates that LDA's supervised projections, which optimize class separability, are far more effective for classification than PCA's variance-maximizing projections.

### Figure 7. Classification Accuracy with LDA (K=3) — Indian Pines Dataset

Line plot of overall classification accuracy versus training size for the three classifiers applied to LDA-reduced Indian Pines data (3 LDs). All three classifiers cluster between ~58–62%, with SVM RBF slightly ahead (~59–62%) and SVM Linear close behind (~59–61%). Compared to PCA (Figure 5), Naive Bayes and SVM Linear show modest improvement (~3–5% higher), while SVM RBF is slightly lower at large training sizes. This suggests that LDA provides a more balanced representation across classifiers, but the overall gain from supervised projection is limited when only 3 dimensions are retained for 16 classes.

### Figure 8. Classification Accuracy without Dimensionality Reduction — Iris Dataset

Line plot of overall classification accuracy versus training size for the three classifiers applied to the full (standardized) 4-feature Iris dataset. SVM Linear (red) leads at ~95–97%, followed by Naive Bayes (blue) at ~94–96%, with SVM RBF (green) at ~92–95%. Compared to PCA (Figure 4, ~88–93%), all classifiers perform notably better without dimensionality reduction, indicating that the 2 discarded PCs carried some useful discriminative information despite contributing only ~4.2% of variance. Compared to LDA (Figure 6, ~96–98%), performance is slightly lower, confirming LDA's superior feature extraction for classification.

### Figure 9. Classification Accuracy without Dimensionality Reduction — Indian Pines Dataset

Line plot of overall classification accuracy versus training size for the three classifiers applied to the full (standardized) 202-band Indian Pines dataset. SVM Linear (red) achieves the highest accuracy, climbing from ~80% at 10% training to ~87% at 50%. SVM RBF (green) follows, rising from ~63% to ~78%. Naive Bayes (blue) performs significantly lower, remaining flat at ~49%. These results are dramatically higher than both PCA (Figure 5, ~55–65%) and LDA (Figure 7, ~58–62%), demonstrating that retaining all 202 spectral bands preserves critical discriminative information that is lost when reducing to only 3 dimensions. SVM Linear outperforms SVM RBF here, suggesting the full-dimensional data is more linearly separable than the compressed representations.

---

## Table

### Table 1. Class-wise Classification Accuracy (Recall) — Indian Pines, PCA (K=3), 30% Training Size

Table listing per-class recall (sensitivity) for each of the 16 Indian Pines land-cover classes under three classifiers (Naive Bayes, SVM Linear, SVM RBF), using PCA-reduced data at 30% training size. Overall macro-average recall is similar across classifiers: Naive Bayes 46.12%, SVM Linear 44.54%, SVM RBF 46.68%. Well-represented classes with distinct spectral signatures achieve high recall — Class 8 (~96–99%), Class 13 (~92–97%), Class 14 (~92–98%), and Class 16 (~86–95%). In contrast, minority and spectrally ambiguous classes suffer severely: Class 9 (0–7%), Class 4 (2–7%), and Class 7 (0–60%) are frequently misclassified. Several classes show 0% recall for specific classifiers (e.g., Class 1 and Class 7 for SVM RBF, Class 9 for SVM Linear and SVM RBF), highlighting that 3 PCs are insufficient to distinguish all 16 land-cover types.
