import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
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

# Dopasowanie modelu liniowego do danych treningowych
X_train = np.array(x_train).reshape(-1, 1)  # Przekształcamy x_train do postaci kolumnowej
Y_train = np.array(y_train)

# Użycie modelu LinearRegression
model = '''do uzupełnienia'''
model.fit(X_train, Y_train)
a = '''do uzupełnienia'''
b = '''do uzupełnienia'''

# Ocena modelu na danych testowych
X_test = np.array(x_test).reshape(-1, 1)
Y_test = np.array(y_test)
y_pred = '''do uzupełnienia'''
precision = '''do uzupełnienia'''#r2_score

# Model wielomianowy stopnia 2
poly_features = '''do uzupełnienia'''#PolynomialFeatures()
X_train_poly = '''do uzupełnienia'''
X_test_poly = '''do uzupełnienia'''
model_poly = LinearRegression()
model_poly.fit(X_train_poly, y_train)

# Ocena modelu wielomianowego na danych testowych
y_pred_poly = '''do uzupełnienia'''
precision_poly = '''do uzupełnienia'''#r2_score

# Wykres prezentujący punkty z danych i dopasowane modele
plt.scatter(x_train, y_train, color='blue', label='Dane treningowe')
plt.scatter(x_test, y_test, color='green', label='Dane testowe')
x_range = np.linspace(min(x_train), max(x_train), 100).reshape(-1, 1)
plt.plot(x_range, model.predict(x_range), color='red', label=f'y = {a:.2f}x + {b:.2f}, r^2={precision:.2f}')
plt.plot(x_range, model_poly.predict(poly_features.transform(x_range)), color='orange', label=f'y = {model_poly.intercept_:.2f} + {model_poly.coef_[1]:.2f}x + {model_poly.coef_[2]:.2f}x^2, r^2={precision_poly:.2f}')
plt.xlabel('Wartość X')
plt.ylabel('Wartość Y')
plt.title('Porównanie modeli')
plt.legend()
plt.savefig('porownanie_modeli.png')
plt.show()