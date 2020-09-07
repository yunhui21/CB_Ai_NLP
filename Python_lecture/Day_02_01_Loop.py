# Day_02_01_Loop.py

# count = input('count:')
# print('Good morning!')



# 문제
# count에 0~3 사이의 숫자를 입력 받아서
# 입력 받은 숫자만큼 아침 인사를 해봇니다.
# 2가지 종류의 코드를 만들어 봅니다.

def not_use():
    count = int(input('count:'))    # 01
    if count == 1:
        print('Good Morning')
    elif count == 2:
        print('Good Morning')
        print('Good Morning')
    elif count == 3:
        print('Good Morning')
        print('Good Morning')
        print('Good Morning')

    count = 1                       #02
    if count > 0:
        print('Good Morning')
        count -= 1
    if count > 0:
        print('Good Morning')
        count -= 1
    if count > 0:
        print('Good Morning')
        count -= 1

# count = 3
# i = 0
# if i < count:
#     print('Good Morning!')
#     i += 1
#     if i < count:
#         print('Good Morning!')
#         i += 1
#         if i < count:
#             print('Good Morning!')
#             i += 1
#             if i < count:
#                 print('Good Morning!')
#                 i += 1

    count = 3
    i = 0
    while i < count:
        print('Good Morning!')
        i += 1

    # 규칙
    # 1 3 5 7 9   range(1, 9, 2)    시작, 종료, 증감
    # 0 1 2 3 4   range(0, 4, 1)
    # 5 4 3 2 1   range(5, 1 ,-1)

    # 규칙 range(1, 9, 2)
    i = 1
    while i <= 9:
        print('hello')
        i += 2
    print('-'*30)
    # 규칙 range (0, 4, 1)
    i = 0
    while i <= 4:
        print('hello')
        i += 1
    print('-'*30)
    # 규칙 range(5, 1 ,-1)
    # i = 5
    # while i <= 1:
    #     print('hello')
    #     i += -1

    i = 5
    while i >= 1:
        print('hello')
        i -= 1

# 문제
# 0~99까지 출력하는 함수를 만드세요.
# 0~99까지 한 줄에 10개씩 출력하는 함수를 만드세요.
# print('hello', end=' ')
# print('python')

def show100():
    # range (0, 99, 1)
    i = 0
    # while i <= 99:
    while i < 100:
        print(i, end=' ')

        if i%10 == 9:
            print()

        i += 1
    # not good
    # i = 0
    # while i < 100:
    #     print(i, end=' ')
    #     i += 1
    #
    #     if i % 10 == 0:
    #         print()

'''
0 1 2 3 4 5 6 7 8 9 
10 11 12 13 14 15 16 17 18 19 
20 21 22 23 24 25 26 27 28 29 
30 31 32 33 34 35 36 37 38 39 
40 41 42 43 44 45 46 47 48 49 
50 51 52 53 54 55 56 57 58 59 
60 61 62 63 64 65 66 67 68 69 
70 71 72 73 74 75 76 77 78 79 
80 81 82 83 84 85 86 87 88 89 
90 91 92 93 94 95 96 97 98 99
'''

show100()

