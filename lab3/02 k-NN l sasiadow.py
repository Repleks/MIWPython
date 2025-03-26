import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier

# Tworzenie rozbudowanego zbioru danych 2D
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [1, 3], [2, 4], [3, 5],
              [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [6, 8], [7, 9], [8, 10]])
y = np.array([0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1])
# Przykładowe etykiety klas (0 lub 1)

# Liczba najbliższych sąsiadów
n_neighbors = 6

# Utworzenie klasyfikatora k-NN
knn = KNeighborsClassifier(n_neighbors=n_neighbors)

# Dopasowanie klasyfikatora do danych
knn.fit(X, y)

# Wizualizacja granic decyzyjnych
h = 0.02  # Rozdzielczość siatki, czyli odległość między punktami na siatce w obu wymiarach
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# określa minimalną i maksymalną wartość cechy w pierwszym wymiarze danych
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
# np.arange(x_min, x_max, h) generuje równomiernie od x_min do x_max z krokiem h, a meshgrid tworzy dwie macierze

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
# przewidywania klas dla każdego punktu na siatce, która została utworzona wcześniej za pomocą funkcji meshgrid.
# xx.ravel() przekształcają macierze w jednowymiarowe tablice
# np.c_łączy te jednowymiarowe tablice w jedną dwuwymiarową tablicę, gdzie każdy wiersz reprezentuje współrzędne jednego punktu na siatce.
# metoda predict klasyfikatora k-NN (knn) jest używana do przewidywania klas dla każdego punktu na siatce na podstawie ich cech.
# Wyniki są zapisywane do tablicy Z, która będzie używana później do wizualizacji granic decyzyjnych na wykresie.

# Tworzenie kolorowej mapy dla obszarów decyzyjnych
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8, 6))

# Wyświetlanie obszarów decyzyjnych
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Wyświetlanie zbioru danych
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title(f'Wizualizacja danych 2D z {n_neighbors} najbliższymi sąsiadami')
plt.colorbar(label='Klasy')
plt.tight_layout()

# Zapisywanie wykresu jako pliku PNG
plt.savefig(f'knn_visualization_{n_neighbors}.png')

# Wyświetlanie wykresu
plt.show()

''' Zadanie na zajęcia - zmień wartość n_neighbors i wyplotuj dla różnych wartości i porównaj wykresy. '''
