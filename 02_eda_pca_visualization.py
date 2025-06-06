import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv("Mall_Customers.csv")

# dropping CustomerID (not useful for clustering)
df = df.drop("CustomerID", axis=1)

# checking for nulls
print("Missing values:\n", df.isnull().sum())

# basic statistics
print("\nSummary statistics:")
print(df.describe())

# countplot for Gender
plt.figure(figsize=(5, 4))
sns.countplot(data=df, x="Gender", palette="pastel")
plt.title("Gender Distribution")
plt.show()

# distribution plots for numerical features
num_cols = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
df[num_cols].hist(bins=15, figsize=(12, 5), color="lightblue", edgecolor="black")
plt.tight_layout()
plt.show()

# PCA for 2D visualisation 
df_numeric = df.drop("Gender", axis=1)
scaled_data = StandardScaler().fit_transform(df_numeric)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

plt.figure(figsize=(6, 5))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c="gray", s=30)
plt.title("PCA Projection of Customers")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)
plt.show()
