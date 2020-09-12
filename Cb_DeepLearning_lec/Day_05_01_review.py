# Day_05_01_review.py
import random


# 문제 1
# 음수와 양수가 섞인 10개짜리 난수 리스트를 만드세요
a = []
for i in range(10):
    a.append(random.randrange(-100, 100))

print(a)

# 문제 2
# 리스트에서 가장 큰 숫자의 위치를 알려주는 함수를 만드세요
def find_max_pos(a, length):
    # length = len(a)
    pos = 0
    for i in range(1, length):
        if a[pos] < a[i]:
            pos = i
            # print(a[i])

    return pos


def selection_sort(a):
    for i in reversed(range(2, len(a) + 1)):
        pos = find_max_pos(a, i)
        a[pos], a[i-1] = a[i-1], a[pos]
        # print(pos)


# pos = find_max_pos(a, len(a))
# print(pos)

# selection_sort(a)
# a.sort()
list.sort(a)
print(a)
