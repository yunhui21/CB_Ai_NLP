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

print_ntimes(3)

# 5)-2 반환값이 있는 함수
def cal_upper(price):
    pass

cal_upper(10000)