import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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

# Dopasowanie modelu liniowego do danych treningowych
X_train = np.array(x_train).reshape(-1, 1)  # Przekształcamy x_train do postaci kolumnowej
Y_train = np.array(y_train)

# Użycie modelu LinearRegression
model = LinearRegression()
model.fit(X_train, Y_train)
a = model.coef_[0]
b = model.intercept_

# Ocena modelu na danych testowych
X_test = np.array(x_test).reshape(-1, 1)
Y_test = np.array(y_test)
y_pred = model.predict(X_test)
precision = r2_score(Y_test, y_pred)

# Wykres prezentujący punkty z danych i dopasowany model liniowy
plt.scatter(x_train, y_train, color='blue', label='Dane treningowe')
plt.scatter(x_test, y_test, color='green', label='Dane testowe')
plt.plot(X_train, model.predict(X_train), color='red', label=f'Model liniowy: y = {a:.2f}x + {b:.2f}')
plt.xlabel('Wartość X')
plt.ylabel('Wartość Y')
plt.title(f'r2_score modelu na danych testowych: {precision:.2f}')
plt.legend()
plt.show()