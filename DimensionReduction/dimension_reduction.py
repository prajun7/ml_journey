# =============================================================================
# CS488/588 Homework 3 - Dimensionality Reduction & Supervised Classification
# =============================================================================
# Datasets:
#   - Iris (loaded via scikit-learn)
#   - Indian Pines (loaded from indianR.mat and indian_gth.mat)
#
# Requirements:
#   pip install numpy matplotlib scipy scikit-learn pandas
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.base import clone
import warnings
warnings.filterwarnings('ignore')

# SECTION 0: DATA LOADING
# ---- Load Iris Dataset (from scikit-learn) -----------------------------------
print("Loading Iris dataset...")
iris = load_iris()
X_iris = iris.data          # Shape: (150, 4)  — 4 spectral features
y_iris = iris.target        # Shape: (150,)    — class labels 0, 1, 2
iris_class_names = iris.target_names
print(f"  Iris:  X={X_iris.shape}, classes={iris_class_names}")

# ---- Load Indian Pines Dataset (from .mat files) ----------------------------
# The .mat file contains:
#   X   -> pixel x band matrix (each row = one pixel sample)
#   gth -> ground truth label per pixel (0 = background, drop these)
print("Loading Indian Pines dataset...")
indian_mat = loadmat('indianR.mat')
gth_mat    = loadmat('indian_gth.mat')

# Extract arrays — variable names per homework spec: X and gth
X_raw = indian_mat['X']          # shape: (bands, pixels) = (202, 21025)
y_raw = gth_mat['gth'].flatten() # 1D ground-truth labels

# If X_raw is 3D (H x W x bands), reshape to 2D (pixels x bands)
if X_raw.ndim == 3:
    H, W, B = X_raw.shape
    X_raw = X_raw.reshape(-1, B)
elif X_raw.shape[0] < X_raw.shape[1]:
    # Data stored as (bands, pixels) — transpose to (pixels, bands)
    X_raw = X_raw.T

# Remove background pixels (label == 0)
mask      = y_raw != 0
X_indian  = X_raw[mask].astype(float)
y_indian  = y_raw[mask]
unique_classes_indian = np.unique(y_indian)
n_classes_indian      = len(unique_classes_indian)
print(f"  Indian Pines: X={X_indian.shape}, {n_classes_indian} classes (background removed)")

# SECTION 0b: STANDARDIZE DATA (zero mean, unit variance — required for PCA/LDA)
scaler_iris   = StandardScaler()
X_iris_scaled = scaler_iris.fit_transform(X_iris)

scaler_indian   = StandardScaler()
X_indian_scaled = scaler_indian.fit_transform(X_indian)

# QUESTION 1a-i  PCA EXPLAINED VARIANCE PLOTS
print("\n[1a-i] Computing PCA explained variance ...")

# ---- PCA on Iris (all 4 components) -----------------------------------------
pca_iris_full = PCA()
pca_iris_full.fit(X_iris_scaled)
ev_iris  = pca_iris_full.explained_variance_ratio_          # individual
cev_iris = np.cumsum(ev_iris)                               # cumulative

# ---- PCA on Indian Pines (all components, display first 30) -----------------
pca_indian_full = PCA()
pca_indian_full.fit(X_indian_scaled)
ev_indian  = pca_indian_full.explained_variance_ratio_
cev_indian = np.cumsum(ev_indian)

# ---- Plot explained variance -------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Iris
pc_labels_iris = [f'PC{i+1}' for i in range(len(ev_iris))]
axes[0].bar(pc_labels_iris, ev_iris * 100, color='steelblue', alpha=0.75,
            label='Individual Variance')
axes[0].plot(pc_labels_iris, cev_iris * 100, 'r-o', linewidth=2,
             markersize=6, label='Cumulative Variance')
axes[0].axhline(y=95, color='green', linestyle='--', linewidth=1.5,
                label='95% Threshold')
axes[0].set_xlabel('Principal Components')
axes[0].set_ylabel('Explained Variance (%)')
axes[0].set_title('Figure 1a. PCA Explained Variance – Iris Dataset')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Indian Pines (first 30 PCs for readability)
n_show = 30
pc_labels_indian = [f'PC{i+1}' for i in range(n_show)]
axes[1].bar(pc_labels_indian, ev_indian[:n_show] * 100, color='darkorange',
            alpha=0.75, label='Individual Variance')
axes[1].plot(pc_labels_indian, cev_indian[:n_show] * 100, 'r-o', linewidth=2,
             markersize=4, label='Cumulative Variance')
axes[1].axhline(y=95, color='green', linestyle='--', linewidth=1.5,
                label='95% Threshold')
axes[1].set_xlabel('Principal Components')
axes[1].set_ylabel('Explained Variance (%)')
axes[1].set_title('Figure 1b. PCA Explained Variance – Indian Pines (First 30 PCs)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].tick_params(axis='x', rotation=70, labelsize=7)

plt.tight_layout()
plt.savefig('Figure1_PCA_ExplainedVariance.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Saved: Figure1_PCA_ExplainedVariance.png")

# ---- Choose K (number of PCs/LDs to retain for subsequent analysis) ----------
# Iris: 2 PCs already capture ~97.7% variance — ideal for visualization
K_iris = 2
K_iris_ev = cev_iris[K_iris - 1] * 100

# Indian Pines: find K that crosses 95% cumulative explained variance
K_indian = int(np.argmax(cev_indian >= 0.95)) + 1
K_indian_ev = cev_indian[K_indian - 1] * 100

print(f"\n  [K Choice] Iris:         K={K_iris}  ({K_iris_ev:.1f}% variance explained)")
print(f"  [K Choice] Indian Pines: K={K_indian} ({K_indian_ev:.1f}% variance explained)")

# QUESTION 1a-ii  PCA 2D VISUALIZATION (first 2 PCs)
print("\n[1a-ii] PCA 2D visualization ...")

pca_2d = PCA(n_components=2)

X_iris_pca2d   = pca_2d.fit_transform(X_iris_scaled)
ev_iris_2d     = pca_2d.explained_variance_ratio_

pca_2d_ip = PCA(n_components=2)
X_indian_pca2d = pca_2d_ip.fit_transform(X_indian_scaled)
ev_indian_2d   = pca_2d_ip.explained_variance_ratio_

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# -- Iris PCA 2D --
colors_iris = plt.cm.Set1(np.linspace(0, 0.5, len(np.unique(y_iris))))
for i, (cls, name) in enumerate(zip(np.unique(y_iris), iris_class_names)):
    m = y_iris == cls
    axes[0].scatter(X_iris_pca2d[m, 0], X_iris_pca2d[m, 1],
                    c=[colors_iris[i]], label=name, alpha=0.8, s=50, edgecolors='k',
                    linewidths=0.3)
axes[0].set_xlabel(f'PC1 ({ev_iris_2d[0]*100:.1f}% var)')
axes[0].set_ylabel(f'PC2 ({ev_iris_2d[1]*100:.1f}% var)')
axes[0].set_title('Figure 2a. PCA 2D Visualization – Iris Dataset')
axes[0].legend(title='Class')
axes[0].grid(True, alpha=0.3)

# -- Indian Pines PCA 2D --
cmap_ip = plt.cm.tab20
for i, cls in enumerate(unique_classes_indian):
    m = y_indian == cls
    axes[1].scatter(X_indian_pca2d[m, 0], X_indian_pca2d[m, 1],
                    color=cmap_ip(i / n_classes_indian),
                    label=f'Class {cls}', alpha=0.5, s=5)
axes[1].set_xlabel(f'PC1 ({ev_indian_2d[0]*100:.1f}% var)')
axes[1].set_ylabel(f'PC2 ({ev_indian_2d[1]*100:.1f}% var)')
axes[1].set_title('Figure 2b. PCA 2D Visualization – Indian Pines Dataset')
axes[1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7,
               title='Class', markerscale=2)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Figure2_PCA_2D_Visualization.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Saved: Figure2_PCA_2D_Visualization.png")

# QUESTION 1a-iii  LDA 2D VISUALIZATION (first 2 LDs)
print("\n[1a-iii] LDA 2D visualization ...")

# LDA n_components <= min(n_classes-1, n_features)
lda_iris_2d   = LDA(n_components=min(2, len(np.unique(y_iris)) - 1))
X_iris_lda2d  = lda_iris_2d.fit_transform(X_iris_scaled, y_iris)

lda_indian_2d  = LDA(n_components=min(2, n_classes_indian - 1))
X_indian_lda2d = lda_indian_2d.fit_transform(X_indian_scaled, y_indian)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# -- Iris LDA 2D --
for i, (cls, name) in enumerate(zip(np.unique(y_iris), iris_class_names)):
    m = y_iris == cls
    ld2 = X_iris_lda2d[m, 1] if X_iris_lda2d.shape[1] >= 2 else np.zeros(m.sum())
    axes[0].scatter(X_iris_lda2d[m, 0], ld2,
                    c=[colors_iris[i]], label=name, alpha=0.8, s=50,
                    edgecolors='k', linewidths=0.3)
axes[0].set_xlabel('LD1')
axes[0].set_ylabel('LD2')
axes[0].set_title('Figure 3a. LDA 2D Visualization – Iris Dataset')
axes[0].legend(title='Class')
axes[0].grid(True, alpha=0.3)

# -- Indian Pines LDA 2D --
for i, cls in enumerate(unique_classes_indian):
    m = y_indian == cls
    axes[1].scatter(X_indian_lda2d[m, 0], X_indian_lda2d[m, 1],
                    color=cmap_ip(i / n_classes_indian),
                    label=f'Class {cls}', alpha=0.5, s=5)
axes[1].set_xlabel('LD1')
axes[1].set_ylabel('LD2')
axes[1].set_title('Figure 3b. LDA 2D Visualization – Indian Pines Dataset')
axes[1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7,
               title='Class', markerscale=2)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Figure3_LDA_2D_Visualization.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Saved: Figure3_LDA_2D_Visualization.png")

# QUESTION 2: CLASSIFICATION SETUP
# Training sizes to evaluate
training_sizes     = [0.10, 0.20, 0.30, 0.40, 0.50]
training_sizes_pct = [int(t * 100) for t in training_sizes]

# Classifiers: Naive Bayes, SVM Linear, SVM RBF
classifiers = {
    'Naive Bayes' : GaussianNB(),
    'SVM (Linear)': SVC(kernel='linear', C=1.0),
    'SVM (RBF)'   : SVC(kernel='rbf',    C=1.0, gamma='scale'),
}

# Fixed visual style across all accuracy plots
plot_styles = {
    'Naive Bayes' : ('-o', 'royalblue'),
    'SVM (Linear)': ('-s', 'tomato'),
    'SVM (RBF)'   : ('-^', 'seagreen'),
}

N_RUNS = 5   # average over multiple random train/test splits for stability

def classify_all_sizes(X, y, classifiers, training_sizes, n_runs=N_RUNS):
    """
    For each training size, run n_runs random splits and average accuracy.
    Returns dict: clf_name -> list of mean accuracies per training size.
    """
    results = {name: [] for name in classifiers}
    for t_size in training_sizes:
        run_accs = {name: [] for name in classifiers}
        for seed in range(n_runs):
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, train_size=t_size, random_state=seed, stratify=y
            )
            for name, clf in classifiers.items():
                model = clone(clf)
                model.fit(X_tr, y_tr)
                run_accs[name].append(accuracy_score(y_te, model.predict(X_te)))
        for name in classifiers:
            results[name].append(np.mean(run_accs[name]) * 100)  # store as %
    return results


def plot_accuracy_vs_training(results, title, filename, training_sizes_pct):
    """Plot overall classification accuracy vs. training size for all classifiers."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, accs in results.items():
        style, color = plot_styles[name]
        ax.plot(training_sizes_pct, accs, style, color=color, label=name,
                linewidth=2, markersize=7)
    ax.set_xlabel('Training Size (%)')
    ax.set_ylabel('Overall Classification Accuracy (%)')
    ax.set_title(title)
    ax.set_xticks(training_sizes_pct)
    ax.set_ylim([0, 105])
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {filename}")

# QUESTION 2a  REDUCE DATA THEN CLASSIFY
# ---- Apply PCA (K components) -----------------------------------------------
pca_K_iris    = PCA(n_components=K_iris)
X_iris_pca    = pca_K_iris.fit_transform(X_iris_scaled)

pca_K_indian  = PCA(n_components=K_indian)
X_indian_pca  = pca_K_indian.fit_transform(X_indian_scaled)

# ---- Apply LDA (min(K, n_classes-1) components) -----------------------------
K_lda_iris    = min(K_iris, len(np.unique(y_iris)) - 1)
lda_K_iris    = LDA(n_components=K_lda_iris)
X_iris_lda    = lda_K_iris.fit_transform(X_iris_scaled, y_iris)

K_lda_indian  = min(K_indian, n_classes_indian - 1)
lda_K_indian  = LDA(n_components=K_lda_indian)
X_indian_lda  = lda_K_indian.fit_transform(X_indian_scaled, y_indian)

print(f"\n[2a] Dimensionality reduction summary:")
print(f"  Iris   PCA: {X_iris.shape[1]}D -> {K_iris}D | "
      f"LDA: {X_iris.shape[1]}D -> {K_lda_iris}D")
print(f"  Indian PCA: {X_indian.shape[1]}D -> {K_indian}D | "
      f"LDA: {X_indian.shape[1]}D -> {K_lda_indian}D")

# CASE i — WITH DIMENSIONALITY REDUCTION
# PCA + Classification (Figure 4 & 5)
print("\n[Case i] PCA + Classification ...")

print("  -> Iris  + PCA")
res_iris_pca   = classify_all_sizes(X_iris_pca,   y_iris,   classifiers, training_sizes)
plot_accuracy_vs_training(
    res_iris_pca,
    f'Figure 4. Classification Accuracy with PCA (K={K_iris}) – Iris Dataset',
    'Figure4_Iris_PCA_Classification.png', training_sizes_pct)

print("  -> Indian Pines + PCA")
res_indian_pca = classify_all_sizes(X_indian_pca, y_indian, classifiers, training_sizes)
plot_accuracy_vs_training(
    res_indian_pca,
    f'Figure 5. Classification Accuracy with PCA (K={K_indian}) – Indian Pines Dataset',
    'Figure5_IndianPines_PCA_Classification.png', training_sizes_pct)

# LDA + Classification (Figure 6 & 7)
print("\n[Case i] LDA + Classification ...")

print("  -> Iris  + LDA")
res_iris_lda   = classify_all_sizes(X_iris_lda,   y_iris,   classifiers, training_sizes)
plot_accuracy_vs_training(
    res_iris_lda,
    f'Figure 6. Classification Accuracy with LDA (K={K_lda_iris}) – Iris Dataset',
    'Figure6_Iris_LDA_Classification.png', training_sizes_pct)

print("  -> Indian Pines + LDA")
res_indian_lda = classify_all_sizes(X_indian_lda, y_indian, classifiers, training_sizes)
plot_accuracy_vs_training(
    res_indian_lda,
    f'Figure 7. Classification Accuracy with LDA (K={K_lda_indian}) – Indian Pines Dataset',
    'Figure7_IndianPines_LDA_Classification.png', training_sizes_pct)

# CASE ii — WITHOUT DIMENSIONALITY REDUCTION (Figure 8 & 9)
print("\n[Case ii] Classification WITHOUT dimensionality reduction ...")

print("  -> Iris (raw features)")
res_iris_raw   = classify_all_sizes(X_iris_scaled,   y_iris,   classifiers, training_sizes)
plot_accuracy_vs_training(
    res_iris_raw,
    'Figure 8. Classification Accuracy without Dim. Reduction – Iris Dataset',
    'Figure8_Iris_NoDR_Classification.png', training_sizes_pct)

print("  -> Indian Pines (raw features)")
res_indian_raw = classify_all_sizes(X_indian_scaled, y_indian, classifiers, training_sizes)
plot_accuracy_vs_training(
    res_indian_raw,
    'Figure 9. Classification Accuracy without Dim. Reduction – Indian Pines Dataset',
    'Figure9_IndianPines_NoDR_Classification.png', training_sizes_pct)

# CASE i  CLASS-WISE ACCURACY TABLE
# Indian Pines, PCA-reduced data, 30% training size, all 3 classifiers
print("\n[Table 1] Class-wise accuracy – Indian Pines, PCA, 30% training ...")

TRAIN_SIZE_TABLE = 0.30   # fixed at 30% per homework requirement

def classwise_recall(X, y, clf, train_size=TRAIN_SIZE_TABLE, n_runs=N_RUNS):
    """
    Return per-class recall (sensitivity) averaged over n_runs splits.
    Dict: class_label (str) -> mean recall
    """
    all_reports = []
    for seed in range(n_runs):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, train_size=train_size, random_state=seed, stratify=y
        )
        model = clone(clf)
        model.fit(X_tr, y_tr)
        report = classification_report(y_te, model.predict(X_te),
                                       output_dict=True, zero_division=0)
        all_reports.append(report)

    classes  = [str(c) for c in np.unique(y)]
    recall   = {}
    for cls in classes:
        vals = [r[cls]['recall'] for r in all_reports if cls in r]
        recall[cls] = np.mean(vals) if vals else np.nan
    return recall

# Gather recall per classifier
table_dict = {}
for name, clf in classifiers.items():
    table_dict[name] = classwise_recall(X_indian_pca, y_indian, clf)

# Build DataFrame (rows = classes, columns = classifiers)
class_labels = [str(c) for c in unique_classes_indian]
df_table = pd.DataFrame(
    {name: [table_dict[name].get(c, np.nan) for c in class_labels]
     for name in classifiers},
    index=[f'Class {c}' for c in class_labels]
)

# Add overall accuracy row (mean recall = macro recall)
overall_row = df_table.mean(axis=0)
overall_row.name = 'Overall (macro)'
df_table = pd.concat([df_table, overall_row.to_frame().T])

# Format as percentages
df_table_pct = (df_table * 100).round(2).astype(str) + '%'

print("\n" + df_table_pct.to_string())

# ---- Save as a matplotlib table figure (Table 1) ----------------------------
fig, ax = plt.subplots(figsize=(10, max(6, len(df_table_pct) * 0.45)))
ax.axis('off')

tbl = ax.table(
    cellText=df_table_pct.values,
    rowLabels=df_table_pct.index,
    colLabels=df_table_pct.columns,
    cellLoc='center',
    loc='center'
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.3, 1.6)

# Highlight header row
for j in range(len(df_table_pct.columns)):
    tbl[0, j].set_facecolor('#4472C4')
    tbl[0, j].set_text_props(color='white', fontweight='bold')

# Highlight overall row
n_rows = len(df_table_pct)
for j in range(len(df_table_pct.columns)):
    tbl[n_rows, j].set_facecolor('#D9E1F2')
    tbl[n_rows, j].set_text_props(fontweight='bold')

ax.set_title(
    'Table 1. Class-wise Classification Accuracy (Recall / Sensitivity)\n'
    f'Indian Pines Dataset – PCA (K={K_indian}), 30% Training Size',
    fontsize=11, fontweight='bold', pad=14
)
plt.tight_layout()
plt.savefig('Table1_Classwise_Accuracy_IndianPines_PCA.png', dpi=150,
            bbox_inches='tight')
plt.show()
print("  Saved: Table1_Classwise_Accuracy_IndianPines_PCA.png")

# SECTION 3: SUMMARY PRINTOUT
print("\n" + "="*65)
print("SUMMARY — Overall accuracy at 30% training size")
print("="*65)
idx30 = training_sizes.index(0.30)  # index for 30% in our list

rows = []
for name in classifiers:
    rows.append({
        'Classifier'        : name,
        'Iris  PCA (%)'     : f"{res_iris_pca[name][idx30]:.1f}",
        'Iris  LDA (%)'     : f"{res_iris_lda[name][idx30]:.1f}",
        'Iris  No DR (%)'   : f"{res_iris_raw[name][idx30]:.1f}",
        'IndPines PCA (%)' : f"{res_indian_pca[name][idx30]:.1f}",
        'IndPines LDA (%)' : f"{res_indian_lda[name][idx30]:.1f}",
        'IndPines No DR(%)': f"{res_indian_raw[name][idx30]:.1f}",
    })

df_summary = pd.DataFrame(rows).set_index('Classifier')
print(df_summary.to_string())

print("\nAll figures and tables saved successfully.")
print("Generated files:")
outputs = [
    "Figure1_PCA_ExplainedVariance.png",
    "Figure2_PCA_2D_Visualization.png",
    "Figure3_LDA_2D_Visualization.png",
    "Figure4_Iris_PCA_Classification.png",
    "Figure5_IndianPines_PCA_Classification.png",
    "Figure6_Iris_LDA_Classification.png",
    "Figure7_IndianPines_LDA_Classification.png",
    "Figure8_Iris_NoDR_Classification.png",
    "Figure9_IndianPines_NoDR_Classification.png",
    "Table1_Classwise_Accuracy_IndianPines_PCA.png",
]
for f in outputs:
    print(f"  • {f}")