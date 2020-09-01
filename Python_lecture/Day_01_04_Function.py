# Day_01_04_Function.py
# 함수 4 종류
# 함수의 핵심 : 데이터 넘겨주고 넘겨받기

# 매개변수 : 교수님께서 나에게 넘겨주는 데이터
# 반환값  : 내게 교수님께 념겨주는 데이터\

# 매기변수 없고, 반환값 없고,
def f_1():
    print('f_1')

f_1()

# 매개변수 있고 - a, 반환값 없고,
def f_2(a, b):     #넘어온 매개변수는 사용을 해주어야 한다. 모두 사용하기
    print('f_2', a, b)

f_2(23, 'abc')

# 매개변수 없고, 반환값 있고.

def f_3():
    # pass      #pass자리에 아무것도 없으면 파이썬은 띄워쓰기로 구분하므로 없으면 에러남
    print('f_3')
    return 17   # 반환값은 사용하지 않을 수 있다. 다른 경우에 사용하기위한 유연한 대체
f_3()


def f_3():
    # pass      #pass자리에 아무것도 없으면 파이썬은 띄워쓰기로 구분하므로 없으면 에러남
    print('f_3')
    return 17   # 반환값은 사용하지 않을 수 있다. 다른 경우에 사용하기위한 유연한 대체
#a = return 17
a = f_3()
print(a)
print(f_3())

# 매개변수 있고, 반환값 있고

def f_4():
    print()
    return

f_4()

# 2자리 정수를 거꾸로 뒤집는 함수로 만드세요.
a = 24
b1 = a//10
b2 = a%10
c = b2*10 + b1

def f_5(n):
    return n%10*10 + n//10

print(f_5(37))
print(f_5(82))


def f_6(a, b):
    if a>=b:
        return a
    else:
        return b
















print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')