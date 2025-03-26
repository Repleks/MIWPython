import numpy as np

# a posteriori = wiarygodność * a priori
# Aktualizacja początkowej wiedzy (a priori) na podstawie nowych danych (wiarygodności),
# co daje nam nową, zaktualizowaną wiedzę (a posteriori).

# W bistro serwują: hamburger, hot-dog, pizza. Oblicz prawdopodobieństwo co jest serwowane.

# Początkowe prawdopodobieństwo stanów (a priori): 
vector = [1 / 3, 1 / 3, 1 / 3]  # Inicjalizacja wektora a priori, każdy posiłek ma równą szansę na wybór.
states = ["Pizza", "HotDog", "Hamburger"]  # Lista dostępnych posiłków.
previous_meal = np.random.choice(states, p=vector)
# Wybór początkowego posiłku na podstawie prawdopodobieństwa a priori.

# a priori to wiedza niezależna od doświadczenia, oparta na aksjomatach/teorii/rozumowaniu, lub czysty strzał.

# Prawdopodobieństwo zaobserwowania danych - wiarygodność.
# W naszym przykładzie w postaci macierzy przejść (transition matrix).
# Czasem znamy macierz przejść w postaci analitycznej, a czasem wynika ona z wielokrotnych iteracji i badań.
transition_matrix = {
    "Pizza": {"Pizza": 0.2, "HotDog": 0.6, "Hamburger": 0.2},
    "HotDog": {"Pizza": 0.3, "HotDog": 0, "Hamburger": 0.7},
    "Hamburger": {"Pizza": 0.5, "HotDog": 0, "Hamburger": 0.5}
}
# Jeśli mamy proces, który jest opisany przez macierz przejść,
# to wektor stacjonarny odpowiada stabilnemu rozkładowi prawdopodobieństwa stanów tego procesu,
# który nie zmienia się w czasie.

meal_counts = [0, 0, 0]  # Inicjalizacja listy liczników dla każdego z posiłków.

# Symulacja procesu wyboru posiłków w bistro.
for _ in range(10_000):
    # Losowanie kolejnego posiłku na podstawie macierzy przejść.
    next_meal = np.random.choice(states, p=[transition_matrix[previous_meal][s] for s in states])

    # Zwiększenie licznika wystąpień danego posiłku.
    meal_counts[states.index(next_meal)] += 1

    # Aktualizacja poprzedniego posiłku.
    previous_meal = next_meal

probability_meal_counts = np.array(meal_counts) / sum(meal_counts)
# Obliczenie prawdopodobieństwa wystąpienia każdego z posiłków na podstawie zebranych danych.
print("Prawdopodobieństwo wystąpienia poszczególnych posiłków =", probability_meal_counts)

# Tworzenie pustej macierzy o wymiarach zgodnych z liczbą dostępnych posiłków.
matrix_size = len(states) # 3
matrix = np.zeros((matrix_size, matrix_size)) # (3, 3)

# Wypełnienie macierzy przejść na podstawie zdefiniowanej macierzy transition_matrix.
for i in range(matrix_size):
    for j in range(matrix_size):
        matrix[i, j] = transition_matrix[states[i]][states[j]]

# Iteracyjne obliczanie wektora stacjonarnego, który odpowiada stabilnemu rozkładowi prawdopodobieństwa stanów procesu.


print("Stacjonarny rozkład prawdopodobieństwa =", vector)

# Obliczenie wektora stacjonarnego dla transition_matrix_computer
eigenvalues, eigenvectors = np.linalg.eig(matrix.T)
stationary_index = np.argmin(np.abs(eigenvalues - 1.0))
stationary_vector = np.real(eigenvectors[:, stationary_index])
stationary_vector /= stationary_vector.sum()
print("stationary_vector (np.linalg.eig)=", stationary_vector)
