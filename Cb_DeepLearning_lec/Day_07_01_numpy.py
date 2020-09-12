# Day_07_01_numpy.py
import numpy as np

# 문제
# 아래 코드를 에러 나지 않게 만드세요
a = [1, 3, 7]
print(np.array(a) + 1)
print(np.int32(a) + 1)

# for i in range(len(a)):
#     a[i] += 1

print(np.zeros(5))
print(np.ones(5))
print(np.ones(5).dtype)
print(np.ones(5, dtype=np.int32))

# 문제
# zeros와 ones를 사용해서 2차원 배열을 만드세요
print(np.zeros([2, 3]))
print(np.zeros([2, 3]).shape)
print(np.ones((2, 3)))

# 문제
# full 함수를 올바르게 호출하세요 (-1로 초기화)
print(np.full((2, 3), -1))

b = np.full((2, 3), -1)

# 문제
# zeros_like 함수를 호출하세요 (b 사용)
print(np.zeros_like(b))
print('-' * 30)

np.random.seed(42)
print(np.random.randn(3))
print(np.random.randn(3, 5))
print(np.random.randint(10, 15, 7))
print(np.random.rand())

c = [3.14, 2.7182, 1.0]
print(np.random.choice(c, 2))
print('-' * 30)

d1 = np.arange(3)
d2 = np.arange(6)
d3 = np.arange(3).reshape(1, 3)
d4 = np.arange(3).reshape(3, 1)
d5 = np.arange(6).reshape(2, 3)

# print(d1 + d2)    # error
print(d1 + d3)      # broadcast
print(d1 + d4)      # broadcast + broadcast
print(d1 + d5)      # broadcast + vector

# print(d2 + d3)    # error
print(d2 + d4)      # broadcast + broadcast
# print(d2 + d5)    # error

print(d3 + d4)      # broadcast + broadcast
print(d3 + d5)      # broadcast + vector

# print(d4 + d5)    # error
print('-' * 30)

# 문제
# 2차원 배열을 거꾸로 출력하세요 (행열 모두)
e = np.arange(12).reshape(3, 4)
print(e)

for i in reversed(e):
    for j in reversed(i):
        print(j, end=' ')
    print()

# 문제
# 2차원 배열을 거꾸로 출력하세요 (슬라이싱 사용)
for i in e[::-1]:
    for j in i[::-1]:
        print(j, end=' ')
    print()
print('-' * 30)

print(e[::-1])
print(e[::-1][::-1])
print(e[::-1, ::-1])

print(e[-1][-1])
print(e[-1, -1])        # fancy indexing

# 문제
# 2차원 배열에서 마지막 열만 출력하세요
# (반복문, 팬시 인덱싱)
for i in range(e.shape[0]):     # 행 갯수 전달
    print(e[i, -1], end=' ')
print()
print(e[:, -1])

print(e)
print(e[-1])
e[-1] = 0
print(e)
print('-' * 30)

# 문제
# 2차원 배열을 만들어서
# 안쪽은 0으로, 바깥쪽 테두리는 1로 채우세요
g = np.zeros([5, 5], dtype=np.int32)
# g[0], g[-1] = 1, 1
# g[:, 0], g[:, -1] = 1, 1
# g[[0, -1]] = 1
g[[0, -1], :] = 1
g[:, [0, -1]] = 1
print(g)

h = np.ones([5, 5], dtype=np.int32)
h[1:-1, 1:-1] = 0
print(h)
print('-' * 30)

i = np.array(range(10))
print(i)
print(i[0], i[3], i[-1])
print(i[[0, 3, -1]])    # index array

# 문제
# 0과 1로 채우는 문제를 인덱스 배열 버전으로 수정하세요

# 문제
# 0으로 채워진 정사각형 배열에 대해
# 양쪽 대각선을 1로 채우세요
j = np.zeros([5, 5], dtype=np.int32)
# j[[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]] = 1
# j[range(5), range(4, -1, -1)] = 1

idx = np.arange(5)
j[idx, idx] = 1
j[idx, idx[::-1]] = 1

print(j)
print('-' * 30)

k = np.arange(12)
np.random.shuffle(k)
print(k)

m = k.reshape(-1, 4)
print(m)

print(np.sum(m))
print(np.sum(m, axis=0))
print(np.sum(m, axis=1))

# 문제
# 2차원 배열에서 가장 큰 값과 가장 작은 값을 구하세요
print(np.max(m, axis=0))
print(np.max(m, axis=1))

print(np.min(m, axis=0))
print(np.min(m, axis=1))
print()

print(np.argmax(m, axis=0))
print('-' * 30)

p = np.int32([2, 1, 5, 3, 8])
q = np.argsort(p)
print(q)            # [1 0 3 2 4]

r = p[q]
print(r)



