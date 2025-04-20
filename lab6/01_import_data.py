# %%
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# %%
# Wczytanie danych z pliku
with open("Dane/dane15.txt", "r") as file:
    data = file.readlines()

# %%
# Podział danych na dane treningowe i testowe
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# %%
# Wykres prezentujący punkty z danych
x_values = []
y_values = []

for line in data:
    x, y = map(float, line.split())  # Zakładamy, że dane są oddzielone spacją
    x_values.append(x)
    y_values.append(y)

# %%
# Wykres
plt.scatter(x_values, y_values, color='blue', label='Dane')
plt.xlabel('Wartość X')
plt.ylabel('Wartość Y')
plt.title('Wykres punktów danych')
plt.legend()
plt.show()