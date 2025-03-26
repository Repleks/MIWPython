import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier  # scikit-learn

# Przykładowy zbiór danych 2D
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [1, 3], [2, 4], [3, 5],
              [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [6, 8], [7, 9], [8, 10]])
y = np.array([0, 0, 1, 0, 1, 0, 1, 2, 1, 2, 0, 1, 1, 1, 1, 1])  # Przykładowe etykiety klas (0 lub 1)

# Utworzenie i dopasowanie klasyfikatora k-NN
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)

# Wizualizacja zbioru danych
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.title('2D Data Visualization with Classes')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Dodanie dodatkowych punktów
X_extra = np.array([[3, 6], [8, 4], [6, 6]])
y_extra = knn.predict(X_extra)
plt.scatter(X_extra[:, 0], X_extra[:, 1], marker='x', c=y_extra, cmap='viridis', label='Additional Points')

plt.legend()
plt.show()

'''Zadanie na zajęcia - zmień wartości niektórych klas 0 i 1 na 2 i zobacz jak zmieni się rezultat'''
