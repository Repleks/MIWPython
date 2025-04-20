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

# Dopasowanie modelu liniowego do danych treningowych
X_train = np.array(x_train).reshape(-1, 1)  # Przekształcamy x_train do postaci kolumnowej
Y_train = np.array(y_train)

# Obliczenie parametrów modelu (a i b) za pomocą metody najmniejszych kwadratów
# Funkcja np.vstack w Pythonie służy do łączenia tablic NumPy wzdłuż osi pionowej, czyli wierszy.
# Funkcja np.linalg.lstsq w bibliotece NumPy służy do rozwiązywania nadokreślonych układów równań liniowych w najmniejszych kwadratach
# rcond jest parametrem, który określa poziom tolerancji dla wyznaczenia odwrotności macierzy
# rcond=None w funkcji np.linalg.lstsq oznacza, że wartość rcond nie jest jawnie określona, co oznacza, że zostanie użyta domyślna wartość. 
A = np.vstack([X_train.T, np.ones(len(X_train))]).T 
a, b = np.linalg.lstsq(A, Y_train, rcond=None)[0]

# Ocena modelu na danych testowych
X_test = np.array(x_test).reshape(-1, 1)
Y_test = np.array(y_test)
y_pred = a*X_test + b
precision = r2_score(Y_test, y_pred)

# Wykres prezentujący punkty z danych i dopasowany model liniowy
plt.scatter(x_train, y_train, color='blue', label='Dane treningowe')
plt.scatter(x_test, y_test, color='green', label='Dane testowe')
plt.plot(X_train, a*X_train + b, color='red', label=f'Model liniowy: y = {a:.2f}x + {b:.2f}')
plt.xlabel('Wartość X')
plt.ylabel('Wartość Y')
plt.title(f'r2_score modelu na danych testowych: {precision:.2f}')
plt.legend()
plt.show()