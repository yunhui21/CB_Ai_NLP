# Day_01_01_basic.py

# ctrl + shift + f10
# alt + 1
# alt + 4
# ctrl + /

# 문제
# hello를 3번 출력하는 코드를 3가지 만드세요
print('hello')
print('hello')
print('hello')

print('hellohellohello')

print('hello' 'hello' 'hello')
print('hello'), print('hello'), print('hello')
print('hello' * 3)
print('hello', 'hello', 'hello')
print()

print('"hello"')
print("'hello'")
print('\'hello\"')
print()

print(12, 3.4, True, 'abc')
print(type(12), type(3.4), type(True), type('abc'))
# <class 'int'> <class 'float'> <class 'bool'> <class 'str'>

a = 7
print(a, 7)

a = 5
# 5 = a
print(a, 7)

print('print :', print)

# a = 7
# b = 3
# a = 7, b = 3
a, b = 7, 3
print(a, b)

# 문제
# a와 b의 값을 교환하세요
# t = a
# a = b
# b = t

a, b = b, a
# a = b
# b = a

print(a, b)

a = 1.23
print(a)
print()

# 연산 : 연달아 하는 계산
# 산술, 관계, 논리, 비트

# 산술 : +  -  *  /  **  //  %
a, b = 7, 3
print(a + b)
print(a - b)
print(a * b)
print(a / b)        # 나눗셈(실수)
print(a ** b)       # 지수
print(a // b)       # 나눗셈(정수)
print(a % b)

#     2             # //
#   +---
# 3 | 7
#     6
#    ---
#     1             # %

# 문제
# 두 자리 양수를 뒤집으세요
# 71 = 7 * 10 + 1
#    = 7 + 1 * 10
a = 71

d1 = a // 10
d2 = a % 10
print(d1 + d2 * 10)

a = d1 + d2 * 10
print(a)

# 문제
# 네 자리 양수를 뒤집으세요
a = 1357

# d1 = a // 1000
# d2 = a // 100 % 10
# d3 = a % 100 // 10
# d4 = a % 10

# d1 = a // 1000 % 10
# d2 = a // 100 % 10
# d3 = a // 10 % 10
# d4 = a // 1 % 10

# print(d1, d2, d3, d4)
# print(d1 * 1 + d2 * 10 + d3 * 100 + d4 * 1000)

a1 = a // 100
a2 = a % 100

a1 = a1 // 10 + a1 % 10 * 10
a2 = a2 // 10 + a2 % 10 * 10

print(a1, a2)
print(a1 + a2 * 100)
print()

print('abc' + 'de')
# print('abc' - 'de')
print('abc' * 3)
print('-' * 130)







