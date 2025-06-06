import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

df = pd.read_csv("Mall_Customers.csv")

df = df.drop("CustomerID", axis=1)

# encoding Gender (Male=1, Female=0)
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])

# scaling features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# fitting KMeans with K=5 clusters (you can change this later)
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_scaled)

# cluster labels
df["Cluster"] = kmeans.labels_

# PCA for 2D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# plotting clusters with PCA components
plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=df["Cluster"], palette="Set2", s=60)
plt.title("K-Means Clusters (K=5) visualized by PCA")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend(title="Cluster")
plt.show()

# number of points per cluster
print("\nCluster counts:")
print(df["Cluster"].value_counts())
