# Day_12_01_numpy.py
import numpy as np

print(np.zeros(3))
print(np.ones(3))
print(np.full(3, fill_value=-1))

print(np.zeros([2, 3]))
print(np.zeros([2, 3], dtype=np.int32))
print('-' * 30)

print(np.random.random(5))
print(np.random.random([2, 3]))     # 0 ~ 1
print(np.random.randn(5))
print(np.random.randn(6).reshape(2, 3))
print(np.random.uniform(0, 10, 5))
print(np.random.random_sample(5))
print(np.random.choice(range(10), 15))
print('-' * 30)

np.random.seed(23)
a = np.random.choice(range(10), 15).reshape(3, 5)
print(a)

print(np.sum(a))
print(np.sum(a, axis=0))    # 열(수직)
print(np.sum(a, axis=1))    # 행(수평)

# 문제
# 2차원 배열에서 가장 큰 값과 작은 값을 찾으세요
print(np.max(a))
print(np.max(a, axis=0))    # 열(수직)
print(np.max(a, axis=1))    # 행(수평)

print(np.min(a))
print(np.min(a, axis=0))    # 열(수직)
print(np.min(a, axis=1))    # 행(수평)

print(np.mean(a))

print(np.argmax(a))
print(np.argmax(a, axis=0))
print(np.argmax(a, axis=1))
print('-' * 50)

print(a)
print(a[0])
print(a[-1])

# a[0] = -1
# print(a)
print()

# 문제
# 2차원 배열을 거꾸로 출력하세요
print(a[::-1])
print(a[::-1][::-1])

b = a[::-1]
c = b[::-1]

print(a[::-1, ::-1])        # fancy indexing

# 문제
# 2차원 배열의 첫 번째와 마지막 번째의 값을 -1로 바꾸세요
a[0][0] = -1
a[-1][-1] = -1
print(a)

a[0, 1] = -2
a[-1, -2] = -2
print(a)

b = a[0]
b[2] = -3
print(a)

# 문제
# 속은 0이고 테두리가 1로 채워진 5행 5열 배열을 만드세요
c = np.zeros([5, 5], dtype=np.int32)
# c[0] = 1
# c[-1] = 1
c[0, :] = 1
c[:, 0] = 1
c[-1, :] = 1
c[:, -1] = 1
print(c)

d = np.ones([5, 5], dtype=np.int32)
d[1:-1, 1:-1] = 0
print(d)

# 문제
# 2차원 배열을 열 순서로 출력하세요 (반복문)
e = np.arange(20).reshape(4, 5)
print(e)
# print(np.transpose(e))

for i in range(e.shape[0]):
    for j in range(e.shape[1]):
        print(e[i, j], end=' ')
        # print('({}, {}) {}'.format(i, j, e[i, j]), end=' ')
    print()

for i in range(e.shape[1]):
    for j in range(e.shape[0]):
        print(e[j, i], end=' ')
    print()

for i in range(e.shape[1]):
    print(e[:, i])
print('-' * 30)

f = np.arange(10)
print(f)

print(f[0], f[1])
print(f[[0, 1]])

g = [1, 5, 3, 2, 3]
print(f[g])             # 인덱스 배열

h = f.reshape(2, 5)
print(h)
print(h[0], h[1])
print(h[[0, 1, 1, 0]])
print()

# 문제
# 단위행렬 (대각선이 1로 채워진 행렬)
# 5행 5열의 단위행렬을 만드세요
print(np.eye(5, dtype=np.int32))

k = np.zeros([5, 5], dtype=np.int32)
# for i in range(len(k)):
#     k[i, i] = 1
# k[0, 0] = k[1, 1] = k[2, 2] = 1
# k[[0, 1, 2], [0, 1, 2]] = 1
k[range(5), range(5)] = 1
print(k)
print('-' * 30)

t = np.int32([1, 3, 4, 9])
bools = [True, False, True, False]

print(t[bools])

for i in range(len(bools)):
    if bools[i]:
        print(t[i])



