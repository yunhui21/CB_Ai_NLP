# Day_02_02_list.py
import random
# colledtion []   ()    {}              <>
#            list tuple set/dictionary  not_used

a = [1, 3, 5]
print(a)
print(a[0], a[1], a[2])

a[0]=90
print(a)


for i in range(len(a)):
    print(a[i])

# 문제
# 100보다 작은 난수 10개로 이루어진 리스트를 반환하는 함수를 만드세요.
def makeRandoms():
    # a = [0, 0, 0, 0, 0, 0, 0, 0 , 0, 0]
    # a = [0]*10
    # for i in range(10):
    #     a[i] = random.randrange(100)

    a = []
    for _ in range(10):
        a.append(random.randrange(100)) #

    return a

random.seed(19)
b = makeRandoms()
print(b)

# a = [1, 3, 5]
# a.append(15)    # 마지막에 새로운 값을 넣어준다.
# print(a)

# 문제
# 리스트를 거꾸로 뒤집는 함수를 만드세요.
# 한번에 값을 2개씩 바꾼다.
# 처음과 마지막
#  0   1   2   3   4   5   6   7   8   9
# [86, 5, 66, 15, 65, 25, 50, 44, 67, 37]
#  37                                 86
#      67                          5

def reverselist(c):
    # a = []
    # for i in range(len(c)-1, -1, -1):
    #     a.append(c[i])
    # for i in range(len(a)):
    #     c[i] = a[i]
    # return a

    for i in range(len(c)//2):
        c[i], c[len(c)-1-i] = c[len(c)-1-i], c[i]


# print(reverselist(b))
reverselist(b)
print(b)

for i in b:     # range, list --> literable
    print(i, end=' ')
print()
# 한번사용하면 무조건 다 써야 한다.
print(type(range(5)), type(b))

for i in reversed(b):
    print(i, end=' ')
print()

for i in reversed(range(len(b))):
    print(b[i], end=' ')
print()

for i in enumerate(b):  # 튜플로 값을 반환, 코딩하다보면 몇번째인 알
    print(i, i[0], i[1])
print()

for i, v in enumerate(b):  # 튜플로 값을 반환, 코딩하다보면 몇번째인 알
    print(i,v)
print()


# tuple은  list의 상수 버전
# t = (1, 2, 3)
# t[0] = 99 튜플은 변환이 불가능
# 잘 사용하지 않는다.
# 튜플은 파이썬이 사용: 매개변수, 반환값을 전달할때

t1, t2 = 1, 2
print(t1, t2)

t3 = 1, 2
print(t3)

# t4, t5 = 1
# print(t4, t5)

a = list(range(0, 10, 2))
print(a)

b = a       # shallow copy(얕은 복사)
c = a.copy()# deep copy(깊은 복사)
a[0] = 99
print(a)
print(b)
print(c)

print('-'*50)
print(a[0])
print(a[-1], a[len(a)-1])
print(a[-2], a[-len(a)])