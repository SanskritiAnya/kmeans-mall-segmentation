import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv("Mall_Customers.csv")

df = df.drop("CustomerID", axis=1)
df["Gender"] = LabelEncoder().fit_transform(df["Gender"])
X_scaled = StandardScaler().fit_transform(df)

# fitting KMeans with any chosen K
k = 5  
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# calculating Silhouette Score
score = silhouette_score(X_scaled, labels)
print(f"Silhouette Score for K={k}: {score:.4f}")
