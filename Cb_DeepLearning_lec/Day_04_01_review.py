# Day_04_01_review.py
import random

print(random.randrange(100))
print(random.randrange(0, 100))
print(random.randrange(0, 100, 1))

# 문제
# 100보다 작은 난수를 10개 출력하세요
for i in range(10):
    print(random.randrange(100), end=' ')
print()

# 문제
# 100보다 작은 난수를 10개 갖는 리스트를 만드세요
a = []
for i in range(10):
    a.append(random.randrange(100))

print(a)

# b = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
b = [0] * 5
for i in range(5):
    b[i] = random.randrange(100)

print(b)

# 문제
# 1차원 리스트를 뒤집으세요
c = []
for i in reversed(range(len(b))):
    # print(b[i])
    c.append(b[i])
print(c)

d = []
for i in range(len(b)):
    d.insert(0, b[i])
print(d)
print()

# 0 9, 1 8, 2 7, 3 6, 4 5
# for i in range(5 // 2):
for i in range(len(b) // 2):
    j = len(b) - 1 - i
    # print(i, j)
    # b[i] = b[j]
    # b[j] = b[i]

    b[i], b[j] = b[j], b[i]
    print(b)

# print(b)

