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


