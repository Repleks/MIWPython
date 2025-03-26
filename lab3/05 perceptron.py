# %%
# import bibliotek
import numpy as np
import matplotlib.pyplot as plt

# %%
# Generowanie losowych danych
np.random.seed(0)
size_of_data = 50
X = np.vstack([
    np.random.normal(loc=[3, 4], scale=[1, 1], size=(size_of_data, 2)),
    np.random.normal(loc=[1, 2], scale=[1, 1], size=(size_of_data, 2))
])
y = np.array([1] * size_of_data + [-1] * size_of_data)

# %%
# Podział danych na zbiór treningowy i testowy
indices = np.random.permutation(len(X))
split = int(0.8 * len(X))
X_train, X_test = X[indices[:split]], X[indices[split:]]
y_train, y_test = y[indices[:split]], y[indices[split:]]

# %%
# Perceptron
n_iterations = 200
learning_rate = 0.1
weights = np.zeros(X.shape[1] + 1)

for _ in range(n_iterations):
    for xi, target in zip(X_train, y_train):
        activation = np.dot(xi, weights[1:]) + weights[0]  # z
        prediction = np.where(activation >= 0, 1, -1)
        error = target - prediction
        weights[1:] += learning_rate * error * xi
        weights[0] += learning_rate * error


# %%
# Dokładność
predictions = np.where(np.dot(X_test, weights[1:]) + weights[0] >= 0, 1, -1)
accuracy = np.mean(predictions == y_test)

# %%
# Wizualizacja granicy decyzyjnej
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
x_values = np.array([np.min(X_train[:, 0]), np.max(X_train[:, 0])])
y_values = -(weights[0] + weights[1] * x_values) / weights[2]
plt.plot(x_values, y_values)
plt.xlabel('Długość płatka')
plt.ylabel('Szerokość płatka')
plt.title(f'Liczba iteracji: {n_iterations}\nDokładność: {accuracy:.2f}')
plt.show()
