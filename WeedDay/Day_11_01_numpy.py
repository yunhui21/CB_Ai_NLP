# Day_11_01_numpy.py
import numpy as np

# 문제
# 0 ~ 9 까지의 정수를 출력하세요 (3가지)
print(np.arange(0, 10, 1))
print(np.arange(0, 10))
print(np.arange(10))

# 배열 : 같은 공간, 같은 자료형
print(type(np.arange(10)))

print(np.arange(-5, 5, 1))
print(np.arange(0, 1, 0.2))
print('-' * 30)

a = np.arange(6)
print(a.shape, a.size, a.ndim)      # (6,) 6 1

# 문제
# reshape을 호출하세요 (2가지)
# b = a.reshape(2, 3)
# b = np.reshape(a, [2, 3])
b = np.reshape(a, (2, 3))
print(b)

print(b.shape, b.size, b.ndim)  # (2, 3) 6 2
print(b.dtype)
print(np.int32)

# 문제
# 2차원 배열을 1차원으로 변환하세요 (3가지)
print(b.reshape(6))
print(b.reshape(len(a)))
print(b.reshape(b.size))
print(b.reshape(b.shape[0] * b.shape[1]))
print(b.reshape(-1))
print('-' * 30)

# 문제
# 1차원 배열을 2차원으로 변환하세요 (3가지)
c = b.reshape(6)

print(c.reshape(2, 3))
print(c.reshape(2, -1))
print(c.reshape(-1, 3))

d = list(range(6))
print(d)

e = np.array(range(6))
print(e)
print(np.array(d))
print(np.array(d).dtype)
print(np.array(d, dtype=np.int8).dtype)
print(np.int64(range(6)))

print('-' * 30)

g = np.arange(6)
h = np.arange(6)

print(g + 1)            # broadcast
print(g ** 2)
print(g > 2)
print(type(g > 2))
print((g > 2).dtype)

print(g + h)            # vector
print(g ** h)
print(g > h)
print('-' * 30)

i = g.reshape(2, 3)
j = h.reshape(2, 3)

print(i + 1)
print(i ** 2)
print(i > 2)

print(i + j)
print(i > j)

print(np.sin(h))            # universal function
print(np.sin(i))
print('-' * 30)

a1 = np.arange(3)
a2 = np.arange(6)
a3 = np.arange(3).reshape(1, 3)
a4 = np.arange(3).reshape(3, 1)
a5 = np.arange(6).reshape(2, 3)

# print(a1 + a2)    # error
print(a1 + a3)      # (3,) => (1, 3)
print(a1 + a4)      # broadcast + broadcast
print(a1 + a5)      # broadcast + vector

# 문제
# a2, a3, a4, a5에 대해서 연산을 해보세요
# print(a2 + a3)    # (1, 6) + (1, 3)
print(a2 + a4)      # (1, 6) + (3, 1)
# print(a2 + a5)    # (1, 6) + (2, 3)

print(a3 + a4)      # (1, 3) +
                    # (3, 1)
print(a3 + a5)      # (1, 3) + (2, 3)

# print(a4 + a5)    # (3, 1) + (2, 3)


print('\n\n\n')