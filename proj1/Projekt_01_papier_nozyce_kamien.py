import random
import numpy as np
import matplotlib.pyplot as plt

# Opis: Program symuluje grę w "Kamień, Papier, Nożyce" pomiędzy graczem a komputerem,
#       aktualizując macierz przejść na podstawie wyników i uczenia się w czasie rzeczywistym.

# Inicjalizacja stanu gotówki gracza
cash = 0
cash_history = [cash]

##### KOMPUTER #####
# Definicja ruchów/taktyki komputera
states_computer = ["Paper", "Rock", "Scissors"]
transition_matrix_computer = {
    "Paper": {"Paper": 2/3, "Rock": 1/3, "Scissors": 0/3},
    "Rock": {"Paper": 0/3, "Rock": 2/3, "Scissors": 1/3},
    "Scissors": {"Paper": 2/3, "Rock": 0/3, "Scissors": 1/3}
}
# Przekształcenie macierzy przejść transition_matrix_computer do postaci tablicy numpy
'''do uzupelnienia'''
matrix_size = len(states_computer)
matrix_computer = np.zeros((matrix_size, matrix_size))
matrix_computer = np.array([[transition_matrix_computer[states_computer[i]][states_computer[j]] for j in range(matrix_size)] for i in range(matrix_size)])
# przejście z matrix zero do matrix wypełnionej wartościami z transition_matrix_computer

# Funkcja wybierająca ruch komputera na podstawie macierzy przejść tj. na podstawie swojego poprzedniego wyboru
def choose_move(player_previous_move):
    index = states_computer.index(player_previous_move)
    return np.random.choice(states_computer, p=matrix_computer[index])
    '''do uzupelnienia'''

##### GRACZ #####
# Definicja ruchów gracza:
#   wersja 1: na podstawie wektora stacjonarnego transition_matrix_computer,
#   wersja 2: w trakcie gry(iteracji) nauczenie gracza taktyki w postaci jego macierzy przejść
#             (inicjujemy macierz przejść gracza wypełnioną np. 1/3, a w trakcie gry po każdej rundzie aktualizujemy ją). 
# Należy napisać kod dla obu wersji (w osobnych plikach, albo w jednym pliku z możliwością zmiany taktyki jakimś parametrem)

# Obliczanie wektora stacjonarnego macierzy przejść transition_matrix_computer (wersja 1 taktyki gracza)
'''do uzupelnienia'''
eigenvalues, eigenvectors = np.linalg.eig(matrix_computer.T)
stationary_index = np.argmin(np.abs(eigenvalues - 1.0))
stationary_vector = np.real(eigenvectors[:, stationary_index])
stationary_vector /= stationary_vector.sum()

# Funkcja aktualizująca macierz przejść gracza (wersja 2 taktyki gracza)
'''do uzupelnienia'''
def update_transition_matrix(player_transition_matrix, previous_move, current_move):
    player_transition_matrix[previous_move][current_move] += 1 # zwiększenie liczby wystąpień przejścia
    total = sum(player_transition_matrix[previous_move].values()) # suma przejść
    for move in player_transition_matrix[previous_move]:
        player_transition_matrix[previous_move][move] /= total # normalizacja przejść
# Funkcja wybierająca ruch gracza na podstawie macierzy przejść tj. na podstawie swojego poprzedniego wyboru (wersja 2 taktyki gracza)
'''do uzupelnienia'''
def choose_player_move(player_transition_matrix, previous_move):
    return np.random.choice(states_computer, p=[player_transition_matrix[previous_move][s] for s in states_computer])


# Główna pętla gry
player_transition_matrix = {
    "Paper": {"Paper": 1/3, "Rock": 1/3, "Scissors": 1/3},
    "Rock": {"Paper": 1/3, "Rock": 1/3, "Scissors": 1/3},
    "Scissors": {"Paper": 1/3, "Rock": 1/3, "Scissors": 1/3}
}
player_previous_move = np.random.choice(states_computer, p=stationary_vector)

version = 2

for _ in range(10000):
    '''do uzupelnienia'''
    computer_move = choose_move(player_previous_move)
    if version ==1:
        player_move = np.random.choice(states_computer, p=stationary_vector)
    if version == 2:
        player_move = choose_player_move(player_transition_matrix, player_previous_move)
        update_transition_matrix(player_transition_matrix, player_previous_move, player_move)

    if (player_move == "Rock" and computer_move == "Scissors") or \
            (player_move == "Scissors" and computer_move == "Paper") or \
            (player_move == "Paper" and computer_move == "Rock"):
        cash += 1
    elif player_move != computer_move:
        cash -= 1

    cash_history.append(cash)
    player_previous_move = player_move

# Wykres zmiany stanu gotówki w każdej kolejnej grze
plt.plot(range(10001), cash_history)
plt.xlabel('Numer Gry')
plt.ylabel('Stan Gotówki')
plt.title('Zmiana Stanu Gotówki w Grze "Kamień, Papier, Nożyce"')
plt.show()