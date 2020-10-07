# 1)Boolean
a = True
print(type(a))
b = False
print(type(b))
# True False은 파이썬의 예약어로 대문자로 표기해야함.

'''
연산자     연산자 의미
==        같다
!=        다르다
>         크다
<         작다
>=        크거나 작다
<=        작거나 같다
'''
print(3==3)
print(3!=3)
print(3<3)
print(3>3)
print(3<=3)
print(3>=3)

mystock = 'naver'
print(mystock == 'naver')

day1 = 10000
day2 = 13000
print((day2-day1) == (day1 * 0.3))

# 2) 논리연산자
'''
기준가격 
1차 계산: 기준 가격에 0.3을 곱한다.'''