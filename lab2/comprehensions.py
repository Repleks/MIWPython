l = [1, 2, 3]
print(l)

l1 = list(range(1, 101)) # 1, 100
print(l1)

l2 = [a * 5 for a in (1, 2, 3)]
print(l2)

# pary każdy z każdym
print([(x, y) for x in range(1, 5) for y in range(4, 0, -1)])

# pary każdy z każdym
print([(x, y) for x in range(1, 6) for y in range(5, 0, -1) if x + y > 7])
