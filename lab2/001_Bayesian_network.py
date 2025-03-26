# Sieć Bayesowska (Bayesian network)
# przykład z wykładu 3 MIW, strona 33 (example from lecture 3 MIW, page 33)
# narysuj tą sieć jako graf skierowany
import numpy as np

# deszczowy dzień (rainy day)
# słoneczny dzień (sunny day)
start = ['rainy day', 'sunny day']
# Prawdopodobieństwo startu (Start probability)
p_start = [0.2, 0.8]

t1 = ['rainy day', 'sunny day']
# Macierz przejść (Transition matrix)
p_t1 = [
    [0.4, 0.6],
    [0.3, 0.7]
]

t2 = ['walk', 'shop', 'clean']
# Macierz emisji (Emission matrix)
p_t2 = [
    [0.1, 0.4, 0.5],
    [0.6, 0.3, 0.1]
]

state = np.random.choice(start, p=p_start, replace=True)

n = 10  # liczba iteracji od startu do czynności (number of iterations from Start to activity)
for i in range(n):
    print(f"Today is {state}, so I should ", end="")
    if state == 'rainy day':
        activity = np.random.choice(t2, p=p_t2[0])
        print(activity)
        state = np.random.choice(t1, p=p_t1[0])
    elif state == 'sunny day':
        activity = np.random.choice(t2, p=p_t2[1])
        print(activity)
        state = np.random.choice(t1, p=p_t1[1])
