import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.datasets import load_iris         
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error 

# LOAD DATA
iris = load_iris()

# Convert to a pandas DataFrame for easier handling
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names) 

print("Dataset shape:", df.shape)
print(df.head())

# QUESTION 1a - i) CORRELATION HEATMAP
plt.figure(figsize=(8, 6))

corr_matrix = df.drop(columns='species').corr() 

# Plot heatmap using seaborn
sns.heatmap(
    corr_matrix,
    annot=True,          
    fmt=".2f",          
    cmap="viridis",
    linewidths=0.5,
    square=True
)

plt.title("1a-i) Iris Feature Correlation Matrix Heatmap")
plt.tight_layout()
plt.savefig("1a_correlation_heatmap.png", dpi=150)
plt.show()

# QUESTION 1a - ii) FEATURE VISUALIZATION (color-coded by species)

# --- Plot A: Pairplot (scatterplot matrix for all feature pairs) ---
# Important: hue='species' colors each species differently
pair_plot = sns.pairplot(df, hue='species', palette='Set1', diag_kind='hist', height=2.2)
pair_plot.fig.suptitle("1a-ii) Iris Pairplot: All Feature Combinations", y=1.02)
plt.savefig("1a_pairplot.png", dpi=150, bbox_inches='tight')
plt.show()

# --- Plot B: Individual feature distributions (histogram + KDE per feature) ---
features = iris.feature_names
colors = ['red', 'green', 'blue']  # One color per species

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()  # Flatten 2x2 grid to loop easily

for i, feature in enumerate(features):
    for j, species in enumerate(iris.target_names):
        # Filter rows belonging to current species
        data = df[df['species'] == species][feature]
        axes[i].hist(data, alpha=0.6, label=species, color=colors[j], bins=15, edgecolor='black')
    
    axes[i].set_title(f"Distribution of {feature}")
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel("Count")
    axes[i].legend()

fig.suptitle("1a-ii) Iris Feature Distributions by Species", fontsize=14)
plt.tight_layout()
plt.savefig("1a_feature_distributions.png", dpi=150)
plt.show()

# QUESTION 2 - LINEAR REGRESSION

# Drop 'petal length' column — this is what we want to PREDICT
# Features used to train: sepal length, sepal width, petal width
X = df.drop(columns=['petal length (cm)', 'species']).values  # Input features (numpy array)
y = df['petal length (cm)'].values                            # Target: petal length

# Pick a test sample index that is NOT in the training set
# We'll use index 73 
sample_index = 73
X_sample = X[sample_index].reshape(1, -1) 
y_actual = y[sample_index]

print(f"\nSample index used for prediction: {sample_index}")
print(f"Actual petal length at index {sample_index}: {y_actual:.4f} cm")


def run_linear_regression(X, y, train_size, sample_index, X_sample, y_actual, case_label):
    """
    Trains a Linear Regression model and evaluates performance.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=42
    )

    # Fit the model on training data
    model = LinearRegression()
    model.fit(X_train, y_train)  # this is where the model learns slope & intercept

    # Predict on test set for RMSE
    y_pred_test = model.predict(X_test)

    # Predict for our specific unknown sample
    y_pred_sample = model.predict(X_sample)[0]  # single-sample prediction

    # RMSE calculation — lower RMSE = better model
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    print(f"Case {case_label}: Train Size = {int(train_size*100)}%")
    print(f"  Slope (coefficients):  {model.coef_}")
    print(f"  Intercept:             {model.intercept_:.4f}")
    print(f"  Predicted petal length (index {sample_index}): {y_pred_sample:.4f} cm")
    print(f"  Actual    petal length (index {sample_index}): {y_actual:.4f} cm")
    print(f"  Prediction Error:      {abs(y_pred_sample - y_actual):.4f} cm")
    print(f"  RMSE (on test set):    {rmse:.4f}")

    # --- Visualization: Actual vs Predicted for test set ---
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred_test, alpha=0.7, color='steelblue', edgecolors='k', label='Test samples')

    # Ideal fit line (y = x means perfect prediction)
    min_val, max_val = min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    # Highlight the specific predicted sample
    plt.scatter(y_actual, y_pred_sample, color='orange', s=150, zorder=5,
                label=f'Index {sample_index} (Actual={y_actual:.2f}, Pred={y_pred_sample:.2f})')

    plt.xlabel("Actual Petal Length (cm)")
    plt.ylabel("Predicted Petal Length (cm)")
    plt.title(f"Q2-{case_label}) Actual vs Predicted | Train={int(train_size*100)}% | RMSE={rmse:.4f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"2_{case_label.lower()}_actual_vs_predicted.png", dpi=150)
    plt.show()

    return model, rmse, y_pred_sample


# CASE i) Train on 30% of data
model_i, rmse_i, pred_i = run_linear_regression(
    X, y, train_size=0.30, sample_index=sample_index,
    X_sample=X_sample, y_actual=y_actual, case_label="i"
)

# CASE ii) Train on 80% of data
model_ii, rmse_ii, pred_ii = run_linear_regression(
    X, y, train_size=0.80, sample_index=sample_index,
    X_sample=X_sample, y_actual=y_actual, case_label="ii"
)


print("COMPARISON SUMMARY")
print(f"  Case i  (30% train) RMSE: {rmse_i:.4f}")
print(f"  Case ii (80% train) RMSE: {rmse_ii:.4f}")