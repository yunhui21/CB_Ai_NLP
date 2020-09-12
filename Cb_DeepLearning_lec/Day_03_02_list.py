# Day_03_02_list.py

# http://210.125.150.125/

# collection : list, tuple, set, dict
#               []    ()     {}    {}

# a, b, c = 1, 3, 5
# print(a, b, c)

a = [1, 3, 5]
print(a)
print(a[0], a[1], a[2])

a[0] = 9
print(a)

print(a[0])
print(a[1])
print(a[2])

i = 0
print(a[i])
i = 1
print(a[i])
i = 2
print(a[i])

a.append(-1)
print(a)

a.pop()
print(a)

for i in range(len(a)):
    # print('**')
    print(a[i], end=' ')
print()

b = [1, 4, 7]
# 문제
# 리스트를 거꾸로 출력하세요
# print(b[2])
# print(b[1])
# print(b[0])

# print(b[len(b) - 1])

for i in range(2, -1, -1):
    print(b[i], end=' ')
print()

for i in range(len(b) - 1, -1, -1):
    print(b[i], end=' ')
print()

# 2 - 0 => 2
# 2 - 1 => 1
# 2 - 2 => 0
for i in range(len(b)):
    # print(len(b) - 1 - i, end=' ')
    print(b[len(b) - 1 - i], end=' ')
print()

for i in reversed(range(len(b))):
    print(b[i], end=' ')
print()

b.reverse()
print(b)

# 문제
# input 함수를 3번 사용해서 입력한 문자열을 리스트에 저장하세요
# c = []
# # c[0] = 12
# for i in range(3):
#     s = input('문자열 : ')
#     # c[i] = s
#     c.append(s)
#
# print(c)

# 문제
# 위의 코드를 함수로 만드세요
def input_texts(count):
    # c = []
    # for i in range(count):
    #     s = input('문자열 : ')
    #     c.append(s)
    #
    # # print(c)
    # return c

    # c = [''] * count
    # for i in range(count):
    #     s = input('문자열 : ')
    #     c[i] = s
    #
    # return c

    return ['sky', 'wind', 'water']


c = input_texts(3)
print(c)


# 문제
# 문자열 리스트에 들어있는 문자의 갯수를 세는 함수를 만드세요
def char_count(words):
    # return len(words[0]) + len(words[1]) + len(words[2])

    total = 0
    for i in range(len(words)):
        # n = len(words[i])
        # # print(n)
        # # total = total + n
        # total += n
        total += len(words[i])

    return total


print(char_count(c))


# 문제
# 리스트에서 가장 큰 숫자를 찾는 함수를 만드세요
def maximum(numbers):
    n = numbers[0]
    # for i in range(len(numbers)):
    for i in range(1, len(numbers)):
        # print(n, numbers[i])
        if n < numbers[i]:
            n = numbers[i]

    return n


# d = [3, 9, 2, 5, 1]
d = [-3, -9, -2, -5, -1]

print(maximum(d))
print(max(d))
print('-' * 30)

# 문제
# 아래처럼 출력하세요

# 0
# 01
# 012
# 0123

s = '0123'
for i in range(len(s)):
    print(s[i], end='')
print()

sum = ''
for i in s:
    sum += i
    print(i, sum)
print()

sum = ''
for i in range(4):
    sum += str(i)
print(sum)





print('\n\n\n')