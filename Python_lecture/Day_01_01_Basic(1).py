#Day_01_01_Basic.py

#문제
#Hello를 3번 출력하세요.
#01
print('hello')
print('hello')
print('hello')

#02
print('hello'*3)

#03
print('hello, hello, hello')

#프로그래밍 : 프로그램을 만드는 과정
#프로그램을 구성하는 요소 : 코드(함수-데이터를 갖다 사용한다), 데이터
#데이터 : 변수(변하는 데이터), 상수(변하지 않는 데이터)
#데이터 : 숫자(정수[integer],실수[float]), 글자(string), 논리값(boolean)

print(3.14, 56, 'hello', True)
print(type(3.14),type(56),type( 'hello'),type( True))

a = 3.14    #대입 연산자 (assignment)
print(a, 3.14)
a = 56
print(a, type(a))

print('hello,\n python!') # newline, 개행문자.
print('"hello"')
print("'hello'")
print('-'*50)


# a = 7 #int a = 7 : 다른언어에서는 지정한후 시작..그래서 파이썬은 타입이 없다라고 말하지만
#       #없는것이 아니다. 다만 바깥으로 들어나지 않는것이다.
# b = 19
a, b = 7, 19
# a = 7, b = 19 파이썬 문법이 아님
# a = 7: b = 19 파이썬 문법이지만 쓰지 않는다.

print(a, b)# integer


# 문제
# 아래쪽 코드에서 거꾸로 출력하도록 코드를 추가해 보세요.
# a 와 b를 서로 교환합니다.
# bug
# a = 19 이렇게 하지 않는건 위에 있는 a, b의 값이 변하지 않고 사용하길 원한다.
# b = 7

#01
# 쥬스, 콜라
# 빈컵 * 쥬스
# 쥬스 * 콜라
# 콜라 * 빈컵

t = a # swap
a = b
b = t # a의 값이 사라지지 않고 t를 통해서 전해진다.
print(a, b)# cpu연산이 빠르다.

#02
a, b = b, a #다중치환 동시에 넣겠다. 01이 연산속도는 빠르다.

print(a, b)

