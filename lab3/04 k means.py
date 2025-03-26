import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Generowanie przykładowego zbioru danych za pomocą funkcji make_blobs
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Inicjalizacja i dopasowanie modelu KMeans z 4 klastrami
kmeans = KMeans(n_clusters=4, n_init=10)
kmeans.fit(X)
# Pobranie współrzędnych centrów klastrów
centers = kmeans.cluster_centers_
# Przypisanie etykiet klastrów dla każdej próbki
labels = kmeans.labels_

# Wizualizacja klastrów i ich centrów na wykresie
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)  # Punkty danych pokolorowane na podstawie przypisanych im klastrów
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, label='Centroids')  # Środki klastrów oznaczone na czerwono
plt.title('K-Means Clustering')  # Tytuł wykresu
plt.xlabel('Feature 1')  # Etykieta osi x
plt.ylabel('Feature 2')  # Etykieta osi y
plt.legend()  # Legenda
plt.show()  # Wyświetlenie wykresu