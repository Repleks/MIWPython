import numpy as np
import matplotlib.pyplot as plt
from logreg_mmajew import LogisticRegression

def softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    softmax_values = exp_x / sum_exp_x
    return softmax_values

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
split = int(0.8 * size_of_data)
X_train = np.array([X[_, :split] for _ in range(len(X))])
X_test = np.array([X[_, split:] for _ in range(len(X))])

# Inicjalizacja i trenowanie modeli regresji logistycznej
models = []  # Inicjalizacja listy do przechowywania modeli
for i in range(4):
    y_train = np.where(np.arange(4) == i, 1, 0).repeat(split) # wektor etykiet klasowych do regresji logistycznej dla klasy i [0, 0, 1, 0] i potem powtórzenie 40 razy
    model = LogisticRegression() # regresja na podstawie tego modelu mmwajew
    model.train(np.vstack(X_train), y_train) # trenowanie stackowanych danych po połączeniu w jedną tablicę
    models.append(model)
'''do uzupelnienia'''


# Przewidywanie klas dla danych testowych i obliczanie dokładności
y_test = np.concatenate([np.full(size_of_data - split, i) for i in range(4)]) # target vector dla danych testowych po połączeniu wszystkich etykiet klasowych z wektorów
y_pred = np.array([model.predict(np.vstack(X_test)) for model in models]).T # przewidywanie klas dla danych testowych transponowana macierz przewidzianych klas
y_pred = np.argmax(y_pred, axis=1) # przewidywane klasy najbardziej prawdopodobne
accuracy = np.mean(y_pred == y_test)
print(f'Accuracy: {accuracy:.2f}')
'''do uzupelnienia'''

# Wyświetlanie punktów danych treningowych i testowych
colors = ['red', 'green', 'blue', 'magenta']
for _ in range(4):
    plt.scatter(X_train[_][:, 0], X_train[_][:, 1], label=f'Class {_}', color=colors[_], marker='o')
    plt.scatter(X_test[_][:, 0], X_test[_][:, 1], color=colors[_], marker='x')

min_x1 = np.min(X[:,:,0])
max_x1 = np.max(X[:,:,0])
min_x2 = np.min(X[:,:,1])
max_x2 = np.max(X[:,:,1])

for _ in range(4):
    [a, b] = models[_].weights
    c = models[_].bias
    # Zakres dla zmiennej x
    x_range = np.array([min_x1, max_x1])
    # Obliczenie wartości zmiennej y na podstawie równania prostej
    y_range = (-a * x_range - c) / b
    # Tworzenie wykresu
    plt.plot(x_range, y_range, color=colors[_])

# Ustawienie limitów dla osi x i y
plt.xlim(min_x1, max_x1)  # Ustawienie limitów dla osi x
plt.ylim(min_x2, max_x2)  # Ustawienie limitów dla osi y

# Wyświetlenie wykresu
plt.savefig('07b logistic regression multiclass_new.png')
plt.show()  # Wyświetlenie wykresu

# Obliczanie softmax dla środkowego punktu wykresu
x1_middle = (min_x1 + max_x1) / 2
x2_middle = (min_x2 + max_x2) / 2
y_prob = np.array([models[_].predict_probability([[x1_middle, x2_middle]]) for _ in range(len(X))]).flatten()
print(f"Prawdopodobieństwa przynależności {x1_middle, x2_middle} reg. log.: {y_prob}")
print(f"Wektor softmax {x1_middle, x2_middle}: {softmax(y_prob)}")

# Obliczanie softmax dla punktu (0,0) na wykresie
y_prob = np.array([models[_].predict_probability([[0, 0]]) for _ in range(len(X))]).flatten()
print(f"Prawdopodobieństwa przynależności {0, 0} reg. log.: {y_prob}")
print(f"Wektor softmax {0, 0}: {softmax(y_prob)}")