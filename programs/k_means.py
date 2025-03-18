import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from load_data import load_dataset
from skimage.transform import resize

# load data
X_train, X_test, y_train, y_test = load_dataset()

# Size reduction to limit features 
X_train = np.array([resize(img, (64, 64)) for img in X_train])

# Images reshaping
X_train = X_train.reshape(X_train.shape[0], -1).astype(np.float32)

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Dimension reduction with PCA
pca = PCA(n_components=50)  # Keeping 50 dimensions to accelerate k-means
X_train_pca = pca.fit_transform(X_train_scaled)

inertia = []
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_train_pca)
    inertia.append(kmeans.inertia_)

#  Curves visualization
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o', linestyle='--')
plt.xlabel("Nombre de clusters (k)")
plt.ylabel("Inertie")
plt.title("Méthode du coude pour choisir k")
plt.show()

# k-means with optimal k  
k_optimal = 2
kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X_train_pca)

# Clusters visualization
plt.figure(figsize=(8, 5))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_kmeans, cmap='viridis', alpha=0.5)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, marker='X', c='red', label='Centroids')
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title(f"Visualisation des clusters (k={k_optimal})")
plt.legend()
plt.show()
