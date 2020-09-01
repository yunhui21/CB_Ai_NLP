# Day_21_02_xor.py


def common(w1, w2, theta, x1, x2):
    value = w1 * x1 + w2 * x2
    return value > theta


def AND(x1, x2):
    return common(1, 1, 1, x1, x2)


def OR(x1, x2):
    return common(1, 1, 0, x1, x2)


def NAND(x1, x2):
    return common(-1, -1, -2, x1, x2)


# 문제
# AND, OR, NAND를 한 번씩 사용합니다
def XOR(x1, x2):
    r1 = OR(x1, x2)
    r2 = NAND(x1, x2)
    return AND(r1, r2)


for x1, x2 in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    # print(AND(x1, x2))
    # print(OR(x1, x2))
    # print(NAND(x1, x2))
    print(XOR(x1, x2))              # F T T F






