import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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

# Przetwarzanie danych treningowych do postaci potrzebnej do dopasowania modelu
X_train = np.array(x_train).reshape(-1, 1)  # Przekształcamy x_train do postaci kolumnowej
Y_train = np.array(y_train)

# Dodanie kolumny x^2 do danych treningowych
# Funkcja np.hstack w bibliotece NumPy służy do łączenia tablic NumPy wzdłuż osi poziomej, czyli kolumn. 
X_train_poly = np.hstack([X_train, X_train**2])

# Obliczenie parametrów modelu wielomianowego (a0, a1, a2) za pomocą metody najmniejszych kwadratów
# Funkcja np.linalg.lstsq w bibliotece NumPy służy do rozwiązywania nadokreślonych układów równań liniowych w najmniejszych kwadratach
A = np.hstack([X_train_poly, np.ones((X_train_poly.shape[0], 1))])
params = np.linalg.lstsq(A, Y_train, rcond=None)[0]

# Przetwarzanie danych testowych
X_test = np.array(x_test).reshape(-1, 1)
Y_test = np.array(y_test)
X_test_poly = np.hstack([X_test, X_test**2])
y_pred = np.dot(X_test_poly, params[:2]) + params[2]  # Obliczenie predykcji modelu wielomianowego
precision = r2_score(Y_test, y_pred)

# Wykres prezentujący punkty z danych i dopasowany model wielomianowy
plt.scatter(x_train, y_train, color='blue', label='Dane treningowe')
plt.scatter(x_test, y_test, color='green', label='Dane testowe')
x_range = np.linspace(min(x_train), max(x_train), 100)
plt.plot(x_range, params[0]*x_range + params[1]*x_range**2 + params[2], color='red', label=f'Model wielomianowy: y = {params[0]:.2f}x^2 + {params[1]:.2f}x + {params[2]:.2f}')
plt.xlabel('Wartość X')
plt.ylabel('Wartość Y')
plt.title(f'r2_score modelu na danych testowych: {precision:.2f}')
plt.legend()
plt.show()