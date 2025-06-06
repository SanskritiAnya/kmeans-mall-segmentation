import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

df = pd.read_csv("Mall_Customers.csv")

df_original = df.copy() 
df = df.drop("CustomerID", axis=1)

le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# retraining KMeans 
kmeans = KMeans(n_clusters=5, random_state=42)
df_original["Cluster"] = kmeans.fit_predict(X_scaled)

# PCA for 2D plotting
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# adding PCA columns
df_original["PCA1"] = X_pca[:, 0]
df_original["PCA2"] = X_pca[:, 1]

# plotting final clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_original, x="PCA1", y="PCA2", hue="Cluster", palette="Set2", s=60)
plt.title("Final Clusters (KMeans) Visualized in 2D (PCA)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster")
plt.grid(True)
plt.show()

# analyzing cluster centers
print("\n Average Values per Cluster:")
cluster_analysis = df_original.groupby("Cluster")[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].mean().round(1)
print(cluster_analysis)
