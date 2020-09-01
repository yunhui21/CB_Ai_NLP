#Day_01_02_Operator.py

#operator 계산, 연산자
#연산자 : 산술, 관계, 논리, 비트
#피타고라스정리 - 함수로 계산

# 산술연산자 : + , - , *, /, //, %, **
a , b = 17, 5

c = a+b #나중에 사용하기위한 방법
print(c)

print(a+b)#다시 사용하지 않아도 되는 방법
print(a-b)
print(a*b)
print(a/b) #소수점이 생성 : 나누어 떨어질때, 나누어 떨어지지 않을때 - cpu가 두가지를 고려하지
            #않도록 처음부터 실수가 되지 않도록 정수연산을 제공한다.
print(a //b) # 3.0에서 추가된 사항: 소수점이하는 버린다. 몫을 구하는식
print(a % b) #나머지를 구하는
print(a**b)# 지수 연산자.

#문제
#2자리 양수를 거꾸로 뒤집어 보세요
# 37 -> 73으로 바꾼다.
#         3
#     +----
#   10|  37
#        30
#     -----
#         7
# 3*10 +7
#a = 7*10 + 3

n = 37
a1 = n//10
a2 = n%10
print(n)

n = a2*10 +a1
print(n%10*10 + n//10)

print('-'*50)

#관계 연산자 : >, >+, <, <=, ==, !=

print(a, b)
print(a > b)
print(a >= b)
print(a < b)
print(a >= b)
print(a == b)
print(a != b)

#ascii code 참고
#항변환 (casting) : int, float, str, bool

print(True)
print(int(True))
print('345')
print(int('345'))
print(int(a != b))#True=1, Flase=0 값을 전환
print(int(False))

#문제
#십대인지를 판단해 보세요.
age = 15 #
b1 = age >= 10 #T , F
b2 = age <= 19 #T , F

# T * T = T
# T * F = F
# F * T = F
# F * F = F

print(bool(b1*b2))
print((age >= 10)*(age <= 19))# 잘 모르면 감싸준다.
print(10<= age <= 19)
print(True * True)# bool인식한다.


#논리 : and  or not : 관계연산자를 연결할때 사용
print(True and True)
print(True and False)
print(False and True)
print(False and False)


print(age >= 10 and age <= 19)# 관계연산이 논리연산보다 우선한다.

#비트연산





