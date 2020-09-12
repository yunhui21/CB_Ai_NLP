# Day_02_03_function.py

# 프로그램 : 코드, 데이터
# 코드 : 함수
# 데이터 : 변수

# 교수 => 데이터 => 학생  : 매개변수
# 교수 <= 데이터 <= 학생  : 반환값


# 매개변수 없고, 반환값 없고.
def f_1():
    print('f_1')

f_1()

# 매개변수 있고, 반환값 없고.
# f_2를 호출하는 2가지 코드를 만드세요
def f_2(a, b):         # a = 3
    print('f_2', a + b)

f_2(3, 8)
f_2(3.14, 8.15)
f_2('3.14', '8.15')

# 매개변수 없고, 반환값 있고.
def f_3():
    print('f_3')

r = f_3()
print(r)

def f_4():
    print('f_4')
    return 11

r = f_4()
print(r)
print(f_4())

# 매개변수 있고, 반환값 있고.
# 문제
# 2개의 숫자 중에서 큰 숫자를 찾는 함수를 만드세요
# max_2
def max_2(a, b):
    # if a > b:
    #     return a
    # else:
    #     return b

    # if a > b:
    #     return a
    # return b

    # if a > b:
    #     b = a
    # return b

    # 문제
    # 조건 연산자를 사용하세요
    return a if a > b else b

print(max_2(3, 7))
print(max_2(7, 3))

# 문제
# 4개의 숫자 중에서 큰 숫자를 찾는 함수를 만드세요
def max_4(a, b, c, d):
    # if a >= b and a >= c and a >= d: return a
    # if b >= a and b >= c and b >= d: return b
    # if c >= a and c >= b and c >= d: return c
    # return d

    # if a < b: a = b
    # if a < c: a = c
    # if a < d: a = d
    # return a

    # 복면가왕
    # return max_2(max_2(a, b), max_2(c, d))

    # 한국시리즈
    return max_2(max_2(max_2(a, b), c), d)


print(max_4(1, 3, 5, 7))
print(max_4(3, 5, 7, 1))
print(max_4(5, 7, 1, 3))
print(max_4(7, 1, 3, 5))




