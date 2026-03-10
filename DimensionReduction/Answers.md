# CS488/588 Homework 3 — Dimensionality Reduction & Supervised Classification

---

## 1a. Dimensionality Reduction Visualizations (30 pts)

### 1a-i) PCA Explained Variance Plots (10 pts)

**Figure 1a. PCA Explained Variance — Iris Dataset.**
Bar chart of the individual explained variance ratio for each of the four principal components of the Iris dataset, overlaid with a cumulative variance curve (red) and a 95% threshold line (green dashed). PC1 accounts for ~73.0% and PC2 for ~22.9% of the total variance. The cumulative line crosses 95% at PC2, reaching approximately 95.8%, which justifies selecting K=2 for subsequent analysis. PC3 and PC4 contribute negligibly (~3.7% and ~0.5%, respectively).

**Figure 1b. PCA Explained Variance — Indian Pines (First 30 PCs).**
Bar chart of individual explained variance (orange bars) for the first 30 of 202 principal components of the Indian Pines hyperspectral dataset, with a cumulative variance curve (red) and a 95% threshold line (green dashed). PC1 alone captures ~70.1% of variance and PC2 adds ~23.5%. The cumulative line crosses the 95% threshold at PC3 (95.63%), so K=3 is selected. The steep drop-off after PC2 — with remaining PCs each contributing less than ~1% — reflects the high spectral redundancy typical of hyperspectral imagery.

### 1a-ii) PCA 2D Visualization (10 pts)

**Figure 2a. PCA 2D Visualization — Iris Dataset.**
Scatter plot of the 150 Iris samples projected onto PC1 (73.0% variance) vs. PC2 (22.9% variance), color-coded by species: setosa (red), versicolor (green), and virginica (orange). Setosa forms a clearly isolated cluster in the upper-left region of the plot, while versicolor and virginica overlap slightly along the PC1 axis, indicating that two PCs effectively reveal the primary class structure of the Iris dataset.

**Figure 2b. PCA 2D Visualization — Indian Pines Dataset.**
Scatter plot of 10,249 Indian Pines pixel samples projected onto PC1 (70.1% variance) vs. PC2 (23.5% variance), color-coded by 16 land-cover classes. Despite capturing ~93.6% of the total variance in two dimensions, significant class overlap is visible throughout the plot. Several classes (e.g., different crop types) cluster together, reflecting the difficulty of separating spectrally similar land-cover types with an unsupervised projection like PCA.

### 1a-iii) LDA 2D Visualization (10 pts)

**Figure 3a. LDA 2D Visualization — Iris Dataset.**
Scatter plot of the Iris samples projected onto LD1 vs. LD2, color-coded by species. Unlike PCA, LDA maximizes between-class separation relative to within-class scatter. All three species are well separated: setosa (red) is fully isolated on the right, versicolor (green) occupies the center, and virginica (orange) clusters on the left. The class separation is markedly improved over PCA (Figure 2a), confirming that LDA is more effective for supervised class discrimination on this dataset.

**Figure 3b. LDA 2D Visualization — Indian Pines Dataset.**
Scatter plot of Indian Pines samples projected onto LD1 vs. LD2, color-coded by 16 classes. Compared to PCA (Figure 2b), LDA leverages label information to produce projections with tighter, more distinguishable clusters for several classes. Some classes (e.g., Class 8, Class 6, Class 14) form more compact groups. However, substantial overlap persists among spectrally similar classes, which is expected given that 16 classes are projected into only two discriminant dimensions.

---

## 1b. Analysis of Dimensionality Reduction Results (5 pts)

### Iris Dataset

**Role of dimensionality reduction:** The Iris dataset has only 4 features, so dimensionality reduction serves primarily as a visualization and feature extraction tool rather than a necessity for computational efficiency. PCA reduces the data from 4D to 2D while retaining 95.8% of the variance, confirming that the original feature space is largely captured by two orthogonal directions.

**Data separability:** In PCA space (Figure 2a), setosa is linearly separable from the other two classes, but versicolor and virginica exhibit partial overlap along both PC1 and PC2. In LDA space (Figure 3a), all three classes achieve clear separation because LDA explicitly optimizes the between-class to within-class scatter ratio. The LDA projection successfully resolves the versicolor–virginica overlap that PCA could not.

**Choice of K:** K=2 is chosen based on the explained variance plot (Figure 1a), where the cumulative variance reaches 95.8% at two components. This is also the maximum number of LDA components for a 3-class problem (n_components ≤ C−1 = 2), so K=2 is used consistently for both PCA and LDA.

**Which method works best:** LDA clearly outperforms PCA for the Iris dataset. The 2D LDA visualization (Figure 3a) shows complete class separation, while PCA (Figure 2a) leaves versicolor and virginica overlapping. This is because PCA is unsupervised — it maximizes total variance without considering class labels — whereas LDA is supervised and directly optimizes class discriminability. For a classification-oriented task, LDA is the superior dimensionality reduction method on this dataset.

### Indian Pines Dataset

**Role of dimensionality reduction:** The Indian Pines dataset has 202 spectral bands with substantial redundancy. PCA reveals that PC1 alone accounts for ~70% of the variance, and 3 PCs capture 95.63%. Dimensionality reduction from 202D to 3D provides significant computational savings and removes correlated noise, but also discards subtle spectral differences among similar land-cover classes.

**Data separability:** PCA's 2D projection (Figure 2b) shows heavy class overlap — most of the 16 land-cover classes are entangled in a dense cloud. This is expected because PCA captures directions of maximum variance (often driven by overall illumination and broad spectral trends), not class-discriminative directions. LDA's 2D projection (Figure 3b) yields visibly tighter clusters for several classes (e.g., Class 8, Class 6, Class 14), but many classes remain mixed. With 16 classes projected into 2D, some overlap is unavoidable.

**Choice of K:** K=3 is chosen based on the 95% cumulative variance threshold from the PCA explained variance plot (Figure 1b). This value is used consistently for both PCA (3 PCs) and LDA (3 LDs). While K=3 captures 95.63% of the variance, it may be aggressive for classification purposes — the discarded 4.37% of variance could contain spectral signatures critical for discriminating spectrally similar crops.

**Which method works best:** LDA provides modestly better 2D visualization than PCA for the Indian Pines dataset, as seen by comparing Figures 2b and 3b. However, neither method achieves strong class separation in 2D due to the inherent complexity of 16 land-cover classes with similar spectral profiles. The benefit of LDA's supervised projection is limited when the number of retained dimensions (K=3) is much smaller than the number of classes (C=16), since LDA can at best produce C−1=15 discriminant directions, and truncating to 3 loses significant class-discriminative information.

---

## 2a. Supervised Classification Visualizations (35 pts)

### 2a-i) Classification with Dimensionality Reduction

**PCA + Classification:**

**Figure 4. Classification Accuracy with PCA (K=2) — Iris Dataset.**
Line plot of overall classification accuracy (%) versus training size (10%–50%) for three classifiers — Naive Bayes (blue), SVM Linear (red), SVM RBF (green) — applied to PCA-reduced Iris data (2 PCs). All three classifiers perform closely together in the ~88–93% range. SVM Linear achieves slightly higher accuracy than the others. Performance is relatively stable across training sizes, indicating that 2 PCs are sufficient for reasonable classification, though some discriminative information is lost compared to using all 4 features.

**Figure 5. Classification Accuracy with PCA (K=3) — Indian Pines Dataset.**
Line plot of overall classification accuracy versus training size for the three classifiers applied to PCA-reduced Indian Pines data (3 PCs). Naive Bayes and SVM Linear remain nearly flat at ~55–56% across all training sizes. SVM RBF stands out, improving from ~57% at 10% training to ~65% at 50% training. The relatively low accuracies reflect severe information loss when compressing 202 spectral bands into only 3 principal components — while these capture 95% of variance, they discard subtle spectral differences needed to discriminate 16 land-cover classes.

**LDA + Classification:**

**Figure 6. Classification Accuracy with LDA (K=2) — Iris Dataset.**
Line plot of overall classification accuracy versus training size for the three classifiers applied to LDA-reduced Iris data (2 LDs). All three classifiers achieve ~96–98% accuracy, which is significantly higher than PCA-based classification (Figure 4, ~88–93%). Naive Bayes and SVM RBF perform nearly identically at ~97–98%, while SVM Linear is marginally lower at ~96%. The marked improvement over PCA demonstrates that LDA's supervised projections, which optimize class separability, are far more effective for classification than PCA's variance-maximizing projections.

**Figure 7. Classification Accuracy with LDA (K=3) — Indian Pines Dataset.**
Line plot of overall classification accuracy versus training size for the three classifiers applied to LDA-reduced Indian Pines data (3 LDs). All three classifiers cluster between ~58–62%, with SVM RBF slightly ahead (~59–62%) and SVM Linear close behind (~59–61%). Compared to PCA (Figure 5), Naive Bayes and SVM Linear show modest improvement (~3–5% higher), while SVM RBF is slightly lower at large training sizes. This suggests that LDA provides a more balanced representation across classifiers, but the overall gain from supervised projection is limited when only 3 dimensions are retained for 16 classes.

### 2a-ii) Classification without Dimensionality Reduction

**Figure 8. Classification Accuracy without Dimensionality Reduction — Iris Dataset.**
Line plot of overall classification accuracy versus training size for the three classifiers applied to the full (standardized) 4-feature Iris dataset. SVM Linear (red) leads at ~95–97%, followed by Naive Bayes (blue) at ~94–96%, with SVM RBF (green) at ~92–95%. Compared to PCA (Figure 4, ~88–93%), all classifiers perform notably better without dimensionality reduction, indicating that the 2 discarded PCs carried some useful discriminative information despite contributing only ~4.2% of variance. Compared to LDA (Figure 6, ~96–98%), performance is slightly lower, confirming LDA's superior feature extraction for classification.

**Figure 9. Classification Accuracy without Dimensionality Reduction — Indian Pines Dataset.**
Line plot of overall classification accuracy versus training size for the three classifiers applied to the full (standardized) 202-band Indian Pines dataset. SVM Linear (red) achieves the highest accuracy, climbing from ~80% at 10% training to ~87% at 50%. SVM RBF (green) follows, rising from ~63% to ~78%. Naive Bayes (blue) performs significantly lower, remaining flat at ~49%. These results are dramatically higher than both PCA (Figure 5, ~55–65%) and LDA (Figure 7, ~58–62%), demonstrating that retaining all 202 spectral bands preserves critical discriminative information that is lost when reducing to only 3 dimensions. SVM Linear outperforms SVM RBF here, suggesting the full-dimensional data is more linearly separable than the compressed representations.

### 2a-iii) Class-wise Accuracy Table

**Table 1. Class-wise Classification Accuracy (Recall / Sensitivity) — Indian Pines, PCA (K=3), 30% Training Size.**
Table listing per-class recall (sensitivity) for each of the 16 Indian Pines land-cover classes under three classifiers (Naive Bayes, SVM Linear, SVM RBF), using PCA-reduced data at 30% training size. Overall macro-average recall is similar across classifiers: Naive Bayes 46.12%, SVM Linear 44.54%, SVM RBF 46.68%. Well-represented classes with distinct spectral signatures achieve high recall — Class 8 (~96–99%), Class 13 (~92–97%), Class 14 (~92–98%), and Class 16 (~86–95%). In contrast, minority and spectrally ambiguous classes suffer severely: Class 9 (0–7%), Class 4 (2–7%), and Class 7 (0–60%) are frequently misclassified. Several classes show 0% recall for specific classifiers (e.g., Class 1 and Class 7 for SVM RBF, Class 9 for SVM Linear and SVM RBF), highlighting that 3 PCs are insufficient to distinguish all 16 land-cover types.

---

## 2b. Analysis of Classification Results (10 pts)

### Iris Dataset

**Role of dimensionality reduction on classification performance:**
Dimensionality reduction has a mixed effect on the Iris dataset depending on the method used:

- **PCA (K=2):** Classification accuracy drops to ~88–93% (Figure 4), which is lower than the no-DR baseline of ~92–97% (Figure 8). Although 2 PCs capture 95.8% of variance, the discarded ~4.2% evidently contains information that contributes to class discrimination — particularly for separating versicolor from virginica. This demonstrates that high explained variance does not guarantee preservation of class-discriminative features.

- **LDA (K=2):** Classification accuracy rises to ~96–98% (Figure 6), which is the highest of all three cases. LDA's supervised projection concentrates class-discriminative information into 2 dimensions more effectively than the original 4-feature space. This confirms that LDA acts as a powerful feature extraction method, not merely a dimensionality reducer.

- **No DR (4 features):** Accuracy is ~92–97% (Figure 8), which falls between PCA and LDA. This is expected — the original features contain all the information, but some of it is diluted across dimensions that are not aligned with class boundaries.

**Data separability:** The Iris dataset is relatively simple (3 classes, 4 features) with setosa being linearly separable and a slight overlap between versicolor and virginica. LDA resolves this overlap by finding the optimal linear projection, leading to the highest accuracy.

**Best method:** LDA + Naive Bayes or LDA + SVM RBF achieves the best performance on Iris (~97–98%). The combination of supervised dimensionality reduction (LDA) with any classifier consistently outperforms both PCA-based and raw-feature approaches. Among the classifiers, Naive Bayes and SVM RBF perform nearly identically under LDA, while SVM Linear is marginally lower. Without dimensionality reduction, SVM Linear performs best (~95–97%), benefiting from the linear separability of the full feature space.

### Indian Pines Dataset

**Role of dimensionality reduction on classification performance:**
Dimensionality reduction has a strongly negative impact on classification accuracy for the Indian Pines dataset:

- **PCA (K=3):** Overall accuracy is only ~55–65% (Figure 5). The macro-average class recall is ~44–47% (Table 1), with several minority classes receiving 0% recall. Compressing 202 bands to 3 PCs based on variance criteria discards spectral nuances essential for distinguishing the 16 land-cover classes. PCA captures broad spectral trends (illumination, vegetation vs. non-vegetation) in the first few components, but the subtle inter-class differences that distinguish, for example, Corn-notill from Corn-mintill reside in lower-variance components.

- **LDA (K=3):** Accuracy improves modestly to ~58–62% (Figure 7). LDA's supervised projection provides ~3–5% improvement over PCA for Naive Bayes and SVM Linear, though SVM RBF gains less. The improvement is limited because 3 discriminant dimensions cannot capture enough of the 15-dimensional LDA subspace (C−1 = 15) needed to separate 16 classes.

- **No DR (202 bands):** Accuracy is dramatically higher — SVM Linear reaches ~80–87% and SVM RBF reaches ~63–78% (Figure 9). This reveals that the spectral information lost during dimensionality reduction is critical for classification. The 202-band feature space, despite its redundancy, preserves subtle spectral signatures that are essential for differentiating similar land-cover types.

**Sensitivity and specificity analysis (Table 1):**
The class-wise recall table for PCA (K=3) at 30% training size reveals severe class imbalance effects:
- Large, spectrally distinct classes achieve high sensitivity: Class 8 (Hay-windrowed, ~96–99%), Class 14 (Woods, ~92–98%), Class 13 (Wheat, ~92–97%).
- Small or spectrally ambiguous classes have very low sensitivity: Class 9 (Oats, 0–7%), Class 4 (Corn-mintill, 2–7%), Class 5 (Grass-pasture, 15–51%), Class 7 (Grass-pasture-mowed, 0–60%).
- Classes with 0% recall are being entirely misclassified into neighboring classes due to the collapse of spectral differences in the 3-PC space. This indicates poor specificity for those classes in the reduced feature space.

**Best method:** Without dimensionality reduction, SVM Linear achieves the best overall accuracy on Indian Pines (~80–87% at varying training sizes). This is a significant finding — for high-dimensional hyperspectral data with many classes, retaining the full feature space coupled with a strong linear classifier outperforms all dimensionality-reduced approaches tested. SVM Linear's strong performance suggests that the 16 Indian Pines classes are approximately linearly separable in the full 202-dimensional space, even though they overlap heavily when projected to 2–3 dimensions.

SVM RBF without DR is the second-best method (~63–78%), and its lower performance relative to SVM Linear may be attributed to the curse of dimensionality — the RBF kernel's distance calculations become less meaningful in very high-dimensional spaces with limited training data.

Naive Bayes without DR performs poorly (~49%) because its conditional independence assumption is strongly violated by the 202 highly correlated spectral bands, leading to unreliable probability estimates.

**Summary:** For the Indian Pines dataset, dimensionality reduction to K=3 (whether by PCA or LDA) sacrifices too much discriminative information for 16 classes. The full 202-band representation with SVM Linear classification is the most effective approach. If dimensionality reduction is required for computational or practical reasons, LDA is preferred over PCA, and a higher K should be considered (e.g., K=10–15) to better balance compression against classification accuracy.

---

## APPENDIX

*See attached Python code: `dimension_reduction.py`*
