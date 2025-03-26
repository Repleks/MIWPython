# Obliczanie prawdopodobieństwa zgodnie z regułą Bayesa
# P(x|Q) = (P(Q|x) * P(x)) / P(Q)

# Dane wejściowe
p_choroba = 0.001  # P(x) Prawdopodobieństwo, że osoba ma chorobę
p_brak_choroby = 1 - p_choroba  # P(-x)=1-P(x) Prawdopodobieństwo, że osoba nie ma choroby
p_test_pozytywny_dla_choroby = 0.99  # P(Q|x) Prawdopodobieństwo pozytywnego wyniku testu, jeśli osoba ma chorobę
p_test_pozytywny_dla_braku_choroby = 1 - p_test_pozytywny_dla_choroby  # P(Q|-x) Prawdopodobieństwo pozytywnego wyniku testu, jeśli osoba nie ma choroby

# Obliczanie P(Q)
# P(Q) Pozytywny wynik testu, niezależnie od stanu zdrowia osoby
# P(Q) to suma:
#   prawdopodobieństwa uzyskania pozytywnego wyniku testu, gdy osoba jest chora, oraz
#   prawdopodobieństwa uzyskania pozytywnego wyniku testu, gdy osoba jest zdrowa
p_test_pozytywny = (p_test_pozytywny_dla_choroby * p_choroba) + (p_test_pozytywny_dla_braku_choroby * p_brak_choroby)
print(f"p_test_pozytywny={round(p_test_pozytywny,2)}")

# Prawdopodobieństwo, że osoba ma chorobę przy pozytywnym wyniku testu
p_choroba_przy_pozytywnym_tescie = (p_test_pozytywny_dla_choroby * p_choroba) / p_test_pozytywny
print(f"p_choroba_przy_pozytywnym_tescie={round(p_choroba_przy_pozytywnym_tescie,2)}")
