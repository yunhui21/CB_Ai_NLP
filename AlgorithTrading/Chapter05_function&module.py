# Chapter05_function&module.py
# 5)-1 function
# print('대신증권')

# for i in range(3):
#     print('대신증권')

# 함수의 입력으로 출력할 '횟수'를 받은후 그 '횟수'만큼 '대신증권'을 출력하는 함수.
# 함수의 입력: 출력 횟수
# 함수의 출력: '대신증권'이라는 문자열(횟수만큼)
# 함수의 동작: 입력 횟수만큼 '대신증권' 문자열을 출력

# def print_ntimes(n):
#     print('대신증권')
#
# print_ntimes(3)
# print_ntimes(2)

def print_ntimes(n):
    for i in range(n):
        print('대신증권')

# print_ntimes(3)

# 5)-2 반환값이 있는 함수
# 5)-2-1 함수 호출 과정 이해하기

def cal_upper(price):
    pass

def cal_upper(price):
    increment = price * 0.3
    upper_price = price + increment
    return upper_price
# cal_upper(10000)
print(cal_upper(10000))

def cal_lower(price):
    decrement = price * 0.3
    lower_price = price - decrement
    return lower_price

print(cal_lower(1000))

# 5)-2-2 두 개의 값 반환하기
def cal_upper_lower(price):
    offset = price * 0.3
    upper = price + offset
    lower = price - offset
    return (upper, lower)

(upper, lower) = cal_upper_lower(10000)
print(upper)
print(lower)

# 5)-3 모듈
# 모듈 : 프로그램이나 하드웨어 기능의 단위'라는 의미를 가짐.
# 모듈은 함수보다 상위 개념으로 함수와 마찬가지로 코드의 재사용을 위해 사용합니다.

# 5)-3-1 모듈 만들기
import stock

print(stock.author)
upper_price = stock.cal_upper(10000)
print(upper_price)
lower_price = stock.cal_lower(10000)
print(lower_price)

if __name__ == '__main__':
    print(cal_upper(10000))
    print(cal_lower(10000))
    print(__name__)

# 5)-3-2 파이썬에서 시간 다루기
# 시계열 : 일정 시간 간격으로 배치된 데이터의 수열
# 시간과 날짜를 다루는 모듈 : time , datetime

import time
print(time.time())
print(time.ctime())
