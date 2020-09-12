# Day_06_01_review.py

# 문제
# 아래처럼 출력하는 함수를 만드세요
#   0123
# 0 *---
# 1 -*--
# 2 --*-
# 3 ---*
def diagonal(stars):
    # for i in range(stars):
    #     for j in range(stars):
    #         if i == j:
    #             print('*', end='')
    #         else:
    #             print('-', end='')
    #     print()

    for i in range(stars):
        print('-' * i, '*', sep='')


diagonal(4)




