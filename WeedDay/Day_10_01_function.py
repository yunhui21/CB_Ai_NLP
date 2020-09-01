# Day_10_01_function.py


def f_1(a, b, c):
    print(a, b, c, end='\n')


f_1(1, 2, 3)            # positional
f_1(a=1, b=2, c=3)      # keyword
f_1(c=3, a=1, b=2)
f_1(1, 2, c=3)
# f_1(1, b=2, 3)        # 키워드는 포지셔널 뒤에.


def f_2(a=0, b=0, c=0):     # default
    print(a, b, c, end='\n')


f_2()
f_2(1)
f_2(1, 2)
f_2(1, b=2, c=3)
f_2(1)
f_2(c=3)


def f_3(*args):             # 가변 인자
    print(args, *args)      # unpacking


# 문제
# f_3을 호출하는 3가지 코드를 만드세요
f_3()
f_3(1)
f_3(1, '2')

a = 1, '2'      # packing
print(a)


def f_4(**kwargs):          # 키워드 가변 인자
    print(kwargs)
    f_5(**kwargs)           # unpacking
    f_5(k=kwargs)


def f_5(**kwargs):
    pass


# 문제
# f_4를 호출하는 3가지 코드를 만드세요
f_4()
f_4(a=1)
f_4(a=1, b=2, c=3)






print('\n\n\n\n\n')
