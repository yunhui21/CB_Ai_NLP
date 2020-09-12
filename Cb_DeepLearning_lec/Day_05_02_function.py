# Day_05_02_function.py


def f_1(a, b, c):
    print(a, b, c)


f_1(1, 2, 3)            # positional
f_1(a=1, b=2, c=3)      # keyword
f_1(c=3, a=1, b=2)
f_1(1, 2, c=3)
# f_1(a=1, 2, c=3)      # keyword는 positional 뒤에.
print()


def f_2(a, b=0, c=0):   # default
    print(a, b, c)


f_2(1)
f_2(1, 2)
f_2(1, 2, 3)
f_2(a=1)
f_2(1, c=3)
print()


def f_3(*args):         # 가변인자
    print(args, *args)  # force unpacking


f_3()
f_3(1)
f_3(1, 'abc')

a = [1, 3, 7]
print(a, *a)
print(a, a[0], a[1], a[2])
print()


# 문제
# f_4를 3가지 방법으로 호출하세요
def f_4(**kwargs):      # keyword 가변인자
    # print(kwargs)
    f_5(**kwargs)
    f_5(k=kwargs)


def f_5(**kwargs):      # keyword 가변인자
    print(kwargs)


f_4()
f_4(a=1)
f_4(a=1, b='abc')
