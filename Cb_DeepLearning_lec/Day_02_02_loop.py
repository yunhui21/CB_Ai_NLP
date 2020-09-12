# Day_02_02_loop.py

# 0 1 2 3 4         0, 4, 1
# 1 3 5 7 9         1, 9, 2
# 4 3 2 1 0         4, 0, -1

# print(0)
# print(1)
# print(2)
# print(3)
# print(4)

# i = 0
# print(i)
# i = 1
# print(i)
# i = 2
# print(i)
# i = 3
# print(i)
# i = 4
# print(i)

for i in range(0, 5, 1):
    print(i, end=' ')
print()

# 문제
# 나머지 2개의 규칙도 반복문으로 구현하세요
for i in range(1, 11, 2):
    print(i, end=' ')
print()

for i in range(5 - 1, -1, -1):
    print(i, end=' ')
print()

for i in range(0, 5, 1):    # 시작, 종료, 증감
    print(i, end=' ')
print()

for i in range(0, 5):       # 시작, 종료, 1
    print(i, end=' ')
print()

for i in range(5):          # 0, 종료, 1
    print(i, end=' ')
print()

print('-' * 30)

# 문제
# 1. 0~100 사이의 숫자를 출력하세요
# 2. 0~25 사이의 숫자를 출력하세요 (한 줄에 5개씩)
for i in range(25):
    print(i, end=' ')

    # if i == 4 or i == 9 or i == 14 or i == 19:
    # if i % 10 == 4 or i % 10 == 9:
    if i % 5 == 4:
        print()
print()

for i in range(25):
    print(i, end='\n' if i % 5 == 4 else ' ')
print()

# 0 1 2 3 4
# 5 6 7 8 9
# 10 11 12 13 14
# 15 16 17 18 19
# 20 21 22 23 24

# 문제
# 아래와 같이 출력하세요

# *
# **
# ***
# ****
# *****

print('*' * 1)
print('*' * 2)
print('*' * 3)
print('*' * 4)
print('*' * 5)

for i in range(5):
    print('*' * i + '*')

print('-' * 30)

# 문제
# kind가 홀수라면 10보다 작은 홀수 전체를
# 짝수라면 10보다 작은 짝수 전체를 출력하세요
# print(1, 3, 5, 7, 9)
kind = '홀수'     # '짝수'
# kind = '짝수'

if kind == '홀수':
    for i in range(1, 10, 2):
        print(i, end=' ')
else:
    for i in range(0, 10, 2):
        print(i, end=' ')
print()

start = 1 if kind == '홀수' else 0

for i in range(start, 10, 2):
    print(i, end=' ')
print()

for i in range(kind == '홀수', 10, 2):
    print(i, end=' ')
print()

for i in range(0, 10, 2):
    print(i + (kind == '홀수'), end=' ')
print()
print('-' * 30)

# 문제
# 반복문을 1개만 써서 아래처럼 만드세요
# *****
# *****
# *****
# *****
for i in range(4):
    # print('*****')
    for j in range(5):
        print('*', end='')
    print()
print()

# for j in range(5):
#     print('*', end='')
# print()

# 문제
# 아래처럼 출력하세요
#
#   0123
# 0 *
# 1 **
# 2 ***
# 3 ****

for i in range(4):
    for j in range(4):
        if i >= j:
            print('*', end='')
        else:
            print('-', end='')
    print()
print()

#   0123
# 3 ****
# 2 ***-
# 1 **--
# 0 *---

for i in reversed(range(4)):
    for j in range(4):
        if i >= j:
            print('*', end='')
        else:
            print('-', end='')
    print()
print()

# ****
# ***
# **
# *

for i in range(4):
    for j in reversed(range(4)):
        if i >= j:
            print('*', end='')
        else:
            print('-', end='')
    print()
print()

#    *
#   **
#  ***
# ****

for i in reversed(range(4)):
    for j in reversed(range(4)):
        if i >= j:
            print('*', end='')
        else:
            print('-', end='')
    print()
print()

# ****
#  ***
#   **
#    *








