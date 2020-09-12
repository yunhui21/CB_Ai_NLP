# Day_05_05_comprehension.py
import random

# 컴프리헨션 : 컬렉션을 만드는 한줄짜리 반복문

a = []
for i in range(5):
    a.append(random.randrange(100))

b = [i for i in range(5)]
b = [random.randrange(100) for i in range(5)]

print(a)
print(b)

# 문제
# 컴프리헨션을 사용해서 1차원 리스트를 복사하세요
c = [i for i in b]
print(c)

# 문제
# 리스트에서 홀수만 뽑아서 리스트를 만드세요
# 리스트에서 가장 큰 홀수를 찾으세요
for i in c:
    if i % 2:
        i

print([i for i in c if i % 2])
print(max([i for i in c if i % 2]))

# 문제
# 2차원 리스트를 1차원으로 변환하세요
# [[56, 83, 23, 77, 2], [88, 27, 21, 51, 7], [16, 98, 95, 64, 14]]
# [56, 83, 23, 77, 2, 88, 27, 21, 51, 7, 16, 98, 95, 64, 14]

a1 = [random.randrange(100) for i in range(5)]
a2 = [random.randrange(100) for i in range(5)]
a3 = [random.randrange(100) for i in range(5)]
d = [a1, a2, a3]
print(d)

print([j for i in d for j in i])
print([0 for i in d])
print([[0] for i in d])

# 문제
# 2차원 리스트에서 가장 큰 값을 찾으세요 (2가지)
print(max([j for i in d for j in i]))
print([max(i) for i in d])
print(max([max(i) for i in d]))

# 문제
# 문자열에 포함된 특정 문자의 갯수를 알려주는 함수를 만드세요
# hello에  l은 2개가 있다
def count(s, ch):
    # 아래 코드를 컴프리헨션으로 바꾸세요
    # cnt = 0
    # for c in s:
    #     # if c == ch:
    #     #     cnt += 1
    #     cnt += (c == ch)
    #
    # return cnt

    return sum([c == ch for c in s])

print(count('hello', 'l'))
print(count('cobalt blue', 'b'))

# 문제 (구글 입사)
# 1 ~ 10000 사이에 포함된 8의 갯수를 구하세요
# 808 => 2
print([str(i) for i in range(10)])
print([count(str(i), '8') for i in range(10)])
print(sum([count(str(i), '8') for i in range(10000)]))
print(sum([str(i).count('8') for i in range(10000)]))

print(list(range(10000)))
print(str(list(range(10000))))
print(str(list(range(10000))).count('8'))
