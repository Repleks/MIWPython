import numpy as np  # Importowanie biblioteki NumPy do operacji na macierzach i wektorach
import matplotlib.pyplot as plt  # Importowanie biblioteki Matplotlib do tworzenia wykresów
from perceptron_mmajew import Perceptron

# Generowanie danych
np.random.seed(0)  # Ustawienie ziarna losowości dla powtarzalności wyników
size_of_data = 50  # Określenie liczby punktów danych w każdej klasie
X = np.array([
    np.random.normal(loc=[1, 1], scale=[1, 1], size=(size_of_data, 2)),  # Generowanie punktów dla pierwszej klasy
    np.random.normal(loc=[10, 10], scale=[1, 2], size=(size_of_data, 2)),  # Generowanie punktów dla drugiej klasy
    np.random.normal(loc=[1, 10], scale=[1, 2], size=(size_of_data, 2)),  # Generowanie punktów dla trzeciej klasy
    np.random.normal(loc=[10, 1], scale=[1, 1], size=(size_of_data, 2))  # Generowanie punktów dla czwartej klasy
])

# Podział danych na zbiór treningowy i testowy
split = int(0.8 * size_of_data)  # Określenie rozmiaru zbioru treningowego na podstawie proporcji
X_train = np.array([X[_, :split] for _ in range(len(X))])  # Podział danych na zbiór treningowy
X_test = np.array([X[_, split:] for _ in range(len(X))])  # Podział danych na zbiór testowy

# Inicjalizacja listy do przechowywania modeli perceptronów i trenowanie ich
models = []

# 1 One-vs-All
# for i in range(4):
#     y_train = np.where(np.arange(4) == i, 1, -1).repeat(split) # wektor etykiet klasowych dla klasy i [-1, -1, 1, -1] i potem powtórzenie 40 razy
#     model = Perceptron(n_iterations=500)
#     model.train(np.vstack(X_train), y_train) # stackowanie tablic w jedną + trenowanie
#     models.append(model)

# 2 One-vs-One
for i in range(4):
    for j in range(i + 1, 4):
        X_train_ij = np.vstack((X_train[i], X_train[j]))
        y_train_ij = np.hstack((np.ones(split), -np.ones(split)))
        model = Perceptron()
        model.train(X_train_ij, y_train_ij)
        models.append((model, i, j))
    '''do uzupelnienia'''

# Przewidywanie klas dla danych testowych i obliczanie dokładności
'''do uzupelnienia'''
# 1 One-vs-All
# y_test = np.concatenate([np.full(size_of_data - split, i) for i in range(4)]) # Wektor etykiet klasowych dla danych testowych jako połączenie etykiet klasowych dla wszystkich wektorów
# y_pred = np.array([model.predict(np.vstack(X_test)) for model in models]).T # Przewidywanie klas dla danych testowych transponowana macierz przewidzianych klas
# y_pred = np.argmax(y_pred, axis=1) # Przewidywane klasy najbardziej prawdopodobne
# accuracy = np.mean(y_pred == y_test)
# print(f'Accuracy: {accuracy:.2f}')

# 2 One-vs-One
y_test = np.concatenate([np.full(size_of_data - split, i) for i in range(4)]) # Wektor etykiet klasowych dla danych testowych jako połączenie etykiet klasowych dla wszystkich wektorów
y_pred = np.array([model.predict(np.vstack(X_test)) for model, _, _ in models]).T # Przewidywanie klas dla danych testowych transponowana macierz przewidzianych klas
y_pred = np.argmax(y_pred, axis=1) # Przewidywane klasy najbardziej prawdopodobne
accuracy = np.mean(y_pred == y_test)
print(f'Accuracy: {accuracy:.2f}')

# Wyświetlanie punktów danych treningowych i testowych
colors = ['red', 'green', 'blue', 'magenta']  # Lista kolorów dla różnych klas
for _ in range(4):
    plt.scatter(X_train[_][:, 0], X_train[_][:, 1], label=f'Class {_}', color=colors[_], marker='o')  # Wyświetlenie punktów treningowych
    plt.scatter(X_test[_][:, 0], X_test[_][:, 1], color=colors[_], marker='x')  # Wyświetlenie punktów testowych

# Rysowanie granic decyzyjnych modeli perceptronów
min_x1 = np.min(X[:,:,0])
max_x1 = np.max(X[:,:,0])
min_x2 = np.min(X[:,:,1])
max_x2 = np.max(X[:,:,1])

for model, _, _ in models:
    [c, a, b] = model.weights  # Współczynniki prostej decyzyjnej
    # Zakres dla zmiennej x
    x_range = np.array([min_x1, max_x1])
    # Obliczenie wartości zmiennej y na podstawie równania prostej
    y_range = (-a * x_range - c) / b
    # Tworzenie wykresu
    plt.plot(x_range, y_range, color=colors[_])  # Rysowanie granic decyzyjnych dla każdej klasy

# Ustawienie limitów dla osi x i y
plt.xlim(min_x1, max_x1)  # Ustawienie limitów dla osi x
plt.ylim(min_x2, max_x2)  # Ustawienie limitów dla osi y

# Zapisanie i wyświetlenie wykresu
plt.savefig('06b perceptron multiclass_new.png')
plt.show()  # Wyświetlenie wszystkich punktów danych oraz granic decyzyjnych