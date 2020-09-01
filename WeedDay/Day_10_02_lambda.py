# Day_10_02_lambda.py


def twice(n):
    return n * 2


def proxy(f, n):
    return f(n)


print(twice)
print(twice(3))

f = twice
print(f(3))

print(proxy(twice, 7))

lam = lambda n: n * 2
print(lam(7))
print(proxy(lam, 7))
print(proxy(lambda n: n * 2, 7))

print('-' * 50)

# 문제
# 1. 리스트를 오름차순으로 정렬하세요
# 2. 리스트를 내림차순으로 정렬하세요
a = [5, 1, 9, 3]

# a.sort()
# list.sort(a)
# print(a)
# b = sorted(a)
# print(a)
# print(b)

print(sorted(a))
print(sorted(a)[::-1])
print(sorted(a, reverse=True))

# 문제
# colors를 오름차순, 내림차순으로 정렬하세요
colors = ['Red', 'green', 'blue', 'YELLOW']
print(sorted(colors))
print(sorted(colors, reverse=True))


def make_lower(s):
    print(s)
    return s.lower()


print(sorted(colors, key=make_lower))
print(sorted(colors, key=str.lower))
print(sorted(colors, key=lambda s: s.lower()))

# 문제
# colors를 길이순으로 정렬하세요 (내림차순 2가지)
print(sorted(colors, key=lambda s: len(s), reverse=True))
print(sorted(colors, key=lambda s: -len(s)))
print(sorted(colors, key=len))


