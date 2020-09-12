# Day_06_02_lambda.py

# 람다 : 반환값을 갖는 한 줄짜리 함수


def proxy_1(f):
    # print(f)
    print(f(3))


def proxy_2(f, n):
    return f(n)


def twice(n):
    return n * 2


l = lambda n: n * 2


t = twice

print(twice(3))
print(t(3))
print(l(3))
print((lambda n: n * 2)(3))
print('twice :', twice, twice(3))
print('-' * 30)

# proxy_1(3)
proxy_1(twice)
result = proxy_2(twice, 7)
result = proxy_2(lambda n: n * 2, 7)
print(result)
print('-' * 30)

a = [31, 29, 57, 43, 78]

# a.sort()
# list.sort(a)

# b = sorted(a)
# print(b)
print(a)
print(sorted(a))


# 문제
# 마지막 자리에 따라 정렬하세요
def last_digit(n):
    print(n)
    return n % 10


# 문제
# 아래 코드를 람다로 바꾸세요
print(sorted(a, key=last_digit))
print(sorted(a, key=lambda n: n % 10))
print('-' * 30)

# 문제
# colors를 길이순으로 정렬하세요 (오름차순)
colors = ['Black', 'Yellow', 'RED', 'blue']

print(sorted(colors))
print(sorted(colors, key=lambda s: s.lower()))
print(sorted(colors, key=lambda s: len(s)))

# 문제
# colors를 길이 역순으로 정렬하세요 (내림차순)
print(sorted(colors, key=lambda s: -len(s)))
print(sorted(colors, key=lambda s: len(s), reverse=True))
