# 🛍️ KMeans Mall Segmentation

This project applies **K-Means Clustering**, an unsupervised machine learning algorithm, to segment mall customers based on age, annual income, and spending score. The goal is to identify distinct customer groups to support targeted marketing strategies.

---

## 📌 Project Overview

- 🔍 Dataset: [Mall Customer Segmentation Data](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial)
- 🧠 Goal: Uncover hidden customer patterns using clustering.
- 🎯 Technique: K-Means Clustering with evaluation via the **Elbow Method** and **Silhouette Score**.

---

## 🛠️ Tech Stack

- Python 3.x  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Seaborn  

---

## 📁 File Structure

| Filename                    | Description                                     |
|----------------------------|-------------------------------------------------|
| `01_load_dataset.py`       | Load and inspect the dataset                    |
| `02_eda_pca_visualization.py` | Perform EDA and reduce features using PCA       |
| `03_kmeans_clustering.py`  | Train KMeans and assign cluster labels          |
| `04_elbow_method.py`       | Determine optimal K using the Elbow Method      |
| `05_cluster_analysis.py`   | Visualize and interpret customer segments       |
| `06_silhouette_score.py`   | Evaluate clustering performance (Silhouette)    |

---

## 📊 Cluster Insights (K = 5)

| Cluster | Age  | Income (k$) | Spending Score | Description                      |
|---------|------|-------------|----------------|----------------------------------|
| 0       | 56.5 | 46.1        | 39.3           | Older, moderate income           |
| 1       | 39.5 | 85.2        | 14.0           | High income, low spending        |
| 2       | 28.7 | 60.9        | 70.2           | Young, good income, high spending |
| 3       | 37.9 | 82.1        | 54.4           | High income, balanced spending   |
| 4       | 27.3 | 38.8        | 56.2           | Young, low income, avg spending  |

---

## 📈 Evaluation

- **Elbow Method**: Suggested optimal `K = 5`
- **Silhouette Score**: `0.2719` → moderate separation between clusters

---

## 🚀 Future Improvements

- Try DBSCAN or Hierarchical Clustering  
- Create a Streamlit dashboard  
- Add gender-wise or region-wise breakdowns  
- Use real-time segmentation for marketing use

---

## 🙋‍♀️ Author

**Sanskriti Anya**  
📍 KIIT University | B.Tech CSE - AI/ML  
🔗 [LinkedIn](https://www.linkedin.com/in/sanskriti-anya-6bb2b4332)



