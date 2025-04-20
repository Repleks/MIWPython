import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Wczytanie danych z pliku
with open("Dane/dane1.txt", "r") as file:
    data = file.readlines()

# Przetwarzanie danych do postaci potrzebnej do dopasowania modelu
x_data = []
y_data = []

for line in data:
    x, y = map(float, line.split())  # Zakładamy, że dane są oddzielone spacją
    x_data.append(x)
    y_data.append(y)

# Podział danych na dane treningowe i testowe
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# Stworzenie cech wielomianowych stopnia 2
poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(np.array(x_train).reshape(-1, 1))
X_test_poly = poly_features.transform(np.array(x_test).reshape(-1, 1))

# Dopasowanie modelu liniowego do danych treningowych
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Predykcja na danych testowych
y_pred = model.predict(X_test_poly)
precision = r2_score(y_test, y_pred)

# Wykres prezentujący punkty z danych i dopasowany model wielomianowy
plt.scatter(x_train, y_train, color='blue', label='Dane treningowe')
plt.scatter(x_test, y_test, color='green', label='Dane testowe')
x_range = np.linspace(min(x_train), max(x_train), 100)
plt.plot(x_range, model.predict(poly_features.transform(x_range.reshape(-1, 1))), color='red', label=f'Model wielomianowy: y = {model.intercept_:.2f} + {model.coef_[1]:.2f}x + {model.coef_[2]:.2f}x^2')
plt.xlabel('Wartość X')
plt.ylabel('Wartość Y')
plt.title(f'r2_score modelu na danych testowych: {precision:.2f}')
plt.legend()
plt.show()