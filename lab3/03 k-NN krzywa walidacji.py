# %%
# biblioteki i dane
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, validation_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

# Wczytanie zbioru danych Iris
iris = load_iris()
X, y = iris.data, iris.target

# %%
# Prezentacja danych
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.title('Iris Data Visualization')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Target')
plt.show()

# %%
# Definicja wartości liczby sąsiadów do sprawdzenia
neighbors = np.arange(1, 20)

# Walidacja krzyżowa dla różnych liczności sąsiadów
cv_scores = []
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')#cv w tej funkcji jest parametrem określającym liczbę podziałów (foldów), 
    cv_scores.append(scores.mean())

# %%
# Wykres walidacji krzyżowej
plt.figure()
plt.plot(neighbors, cv_scores, marker='o')
plt.title('Cross-Validation Scores for Different Number of Neighbors')
plt.xlabel('Number of Neighbors')
plt.ylabel('Mean CV Accuracy')
plt.xticks(neighbors)
plt.grid(True)
plt.show()

# %%
# Krzywa walidacji dla liczby sąsiadów
train_scores, test_scores = validation_curve(KNeighborsClassifier(), X, y, param_name="n_neighbors", param_range=neighbors, cv=5, scoring="accuracy")
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

#%%
# Wykres krzywej walidacji
plt.figure()
plt.plot(neighbors, train_mean, label="Training score", color="darkorange", marker='o')
plt.fill_between(neighbors, train_mean - train_std, train_mean + train_std, alpha=0.2, color="darkorange")
plt.plot(neighbors, test_mean, label="Cross-validation score", color="navy", marker='o')
plt.fill_between(neighbors, test_mean - test_std, test_mean + test_std, alpha=0.2, color="navy")
plt.title("Validation Curve with k-NN")
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
plt.xticks(neighbors)
plt.legend(loc="best")
plt.grid(True)
plt.show()

'''Zadanie na zajęcia : przeanalizują walidację krzyżową'''