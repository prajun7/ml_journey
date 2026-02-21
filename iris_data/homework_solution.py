import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# --- Part 1: Iris Data Visualization ---

# Load the Iris dataset
iris = load_iris() 
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target_names[iris.target]

print("\n--- Part 1: Iris Data Visualization ---")
print("First 5 rows of the Iris dataset:\n", iris_df.head())

# 1a) i) Correlation coefficient - display correlation matrix as heat map
print("\n1a) i) Correlation Matrix Heatmap")
plt.figure(figsize=(8, 6))
sns.heatmap(iris_df.iloc[:, :-1].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Iris Features')
plt.savefig('correlation_heatmap.png')
plt.show()

# 1a) ii) Feature analysis/visualization - display all the color coded features
print("\n1a) ii) Feature Analysis/Visualization (Pair Plots)")
sns.pairplot(iris_df, hue='species', palette='viridis')
plt.suptitle('Pair Plot of Iris Features by Species', y=1.02)
plt.savefig('pair_plot.png')
plt.show()

# 1b) Data analysis based on the visualizations
print("\n1b) Data Analysis based on Visualizations")
print("\nImplications of data or feature distribution on data analysis:")
print("From the pair plots, we can observe distinct clusters for different species, especially when considering petal length and petal width. Setosa is clearly separable from Versicolor and Virginica. Versicolor and Virginica show some overlap, but are generally distinguishable. This suggests that these features are highly informative for classifying Iris species. The distributions of features like petal length and width are crucial for separating the species.")
print("\nInferences from 1a) i) and ii) - data patterns/trends and their influence on data analysis:")
print("The correlation heatmap shows strong positive correlations between petal length and petal width, and also between sepal length and petal length/width. This indicates that as one of these features increases, the others tend to increase as well. For example, longer petals tend to be wider. This strong correlation can be useful for predicting one feature from another, as seen in the linear regression task. The distinct clustering in the pair plots confirms that species can be effectively differentiated using these features, making classification tasks relatively straightforward for this dataset.")

# --- Part 2: Linear Regression (LR) analysis using sklearn ---

print("\n--- Part 2: Linear Regression (LR) Analysis ---")

# Drop 'petal length' feature and prepare data
X = iris_df.drop(['petal length (cm)', 'species'], axis=1)
y = iris_df['petal length (cm)']

# Case i) 30% samples for training (train_size = 0.3)
print("\nCase i) 30% samples for training")
X_train_30, X_test_30, y_train_30, y_test_30 = train_test_split(X, y, train_size=0.3, random_state=42)

lr_model_30 = LinearRegression()
lr_model_30.fit(X_train_30, y_train_30)

y_pred_30 = lr_model_30.predict(X_test_30)

# Output LR parameters
print("  LR Parameters (30% train):")
print("    Coefficients (Slope):", lr_model_30.coef_)
print("    Intercept:", lr_model_30.intercept_)

# Predict for an unknown sample X (random sample index from test set)
# We need to ensure the sample is from the original dataset but not in the training set.
# A simple way is to pick an index from the X_test_30 set.
random_test_index_30 = X_test_30.index[0]
unknown_sample_30 = X.loc[[random_test_index_30]]
predicted_petal_length_30 = lr_model_30.predict(unknown_sample_30)
actual_petal_length_30 = y.loc[random_test_index_30]

print(f"  Prediction for sample at index {random_test_index_30} (30% train):")
print(f"    Actual Petal Length: {actual_petal_length_30:.2f} cm")
print(f"    Predicted Petal Length: {predicted_petal_length_30[0]:.2f} cm")

# Quantitative performance analysis (RMSE)
rmse_30 = np.sqrt(mean_squared_error(y_test_30, y_pred_30))
print(f"  RMSE (30% train): {rmse_30:.2f}")

# Case ii) 80% samples for training (train_size = 0.8)
print("\nCase ii) 80% samples for training")
X_train_80, X_test_80, y_train_80, y_test_80 = train_test_split(X, y, train_size=0.8, random_state=42)

lr_model_80 = LinearRegression()
lr_model_80.fit(X_train_80, y_train_80)

y_pred_80 = lr_model_80.predict(X_test_80)

# Output LR parameters
print("  LR Parameters (80% train):")
print("    Coefficients (Slope):", lr_model_80.coef_)
print("    Intercept:", lr_model_80.intercept_) 

# Predict for an unknown sample X (random sample index from test set)
random_test_index_80 = X_test_80.index[0]
unknown_sample_80 = X.loc[[random_test_index_80]]
predicted_petal_length_80 = lr_model_80.predict(unknown_sample_80)
actual_petal_length_80 = y.loc[random_test_index_80]

print(f"  Prediction for sample at index {random_test_index_80} (80% train):")
print(f"    Actual Petal Length: {actual_petal_length_80:.2f} cm")
print(f"    Predicted Petal Length: {predicted_petal_length_80[0]:.2f} cm")

# Quantitative performance analysis (RMSE)
rmse_80 = np.sqrt(mean_squared_error(y_test_80, y_pred_80))
print(f"  RMSE (80% train): {rmse_80:.2f}")

# Analysis of which case was better
print("\nAnalysis: Which case was better and why?")
print(f"  RMSE for 30% training data: {rmse_30:.2f}")
print(f"  RMSE for 80% training data: {rmse_80:.2f}")

if rmse_30 < rmse_80:
    print("  Case i) with 30% training data resulted in a lower RMSE, indicating a better model performance on the test set. This is unexpected as typically more training data leads to better models. This could be due to the random split of the data, or the specific characteristics of the Iris dataset where a smaller, representative training set might generalize well.")
else:
    print("  Case ii) with 80% training data resulted in a lower RMSE, indicating a better model performance on the test set. This is generally expected as a larger training set allows the model to learn more patterns and generalize better to unseen data. A lower RMSE means the predictions are closer to the actual values.")

print("\nVisualizing Actual vs. Predicted Petal Length (30% train):")
plt.figure(figsize=(10, 6))
plt.scatter(y_test_30, y_pred_30, alpha=0.7) 
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Actual Petal Length (cm)')
plt.ylabel('Predicted Petal Length (cm)')
plt.title('Actual vs. Predicted Petal Length (30% Train)')
plt.grid(True)
plt.savefig('actual_vs_predicted_30_train.png')
plt.show()

print("\nVisualizing Actual vs. Predicted Petal Length (80% train):")
plt.figure(figsize=(10, 6))
plt.scatter(y_test_80, y_pred_80, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Actual Petal Length (cm)')
plt.ylabel('Predicted Petal Length (cm)')
plt.title('Actual vs. Predicted Petal Length (80% Train)')
plt.grid(True)
plt.savefig('actual_vs_predicted_80_train.png')
plt.show()
