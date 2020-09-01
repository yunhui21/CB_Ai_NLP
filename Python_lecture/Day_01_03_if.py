# Day_01_03_if.py

a = 13
print(a%2) #나누어 떨어지거나 아니거나.
#두 가지 모두 출력되는건 오류이므로 조건에 따른 선택 출력이 이루어지도록 해야 한다.
if a%2 == 1: #:이 나오면 들여쓰기
    print('홀수')
else:
    print('짝수')

if a%2 :
    print('홀수')
else:
    print('짝수')

if a:
    print('홀수')
else:
    print('짝수')

if a:
    print('홀수')

#문제
#0~999 사이의 숫자를 입력 받아서
# 몇 자리 숫자인지 맞춰 보세요.

a = 4 # int(input('number:')) #inter변환
print(a)
print(type(a))

#01
if a >= 100:
     print(3)
if 100 > a >= 10:
     print(2)
if 10 > a >= 0:
     print(1)
# 02
if a >= 100:
     print(3)
else:
    if a >= 10:
        print(2)
    else:
        print(1)
# 03
a1, a2, a3 = bool(a//100), bool(a//10), bool(a//1) #쉽게 보여지지 않으면 수행한후 검증이 필요할수있다.
print(a1 + a2 + a3)

# 04
digit = 0

# if a >= 100:
#     digit = digit + 1
#     a = a // 10
# if a >= 10:
#     digit = digit +1
#     a = a // 10

# if a >= 100:    digit = digit + 1;    a = a // 10
# if a >=  10:    digit = digit + 1;    a = a // 10

if a >=  10:    digit = digit + 1;    a = a // 10
if a >=  10:    digit = digit + 1;    a = a // 10

print(digit)

#applekoong@naver.com
#이름을 적어서 비어있는 메일을 보내세요.

print(


    'hello'

)
#파이썬 인터프리터가 해석하지 않는것. [공백] : space, return, tap
#조건의 개수만큼 if문을 만들어야 한다. 개발자는 알고리즘으로 넣어버린다.
if a >= 100:
    print(3)
elif a >= 10:
    print(2)
else:
    print(1)

print('end')
# 공백 : space, return, tab


# 문제
# 2개의 정수 중에서 큰 숫자를 찾는 함수를 만드세요. 함수안에서 출력하면 안된다.
# 힌트 : 복명가왕, 한국시리즈

# 복면가왕
def max2(a, b):
    # if a >= b:
    #     return a    # return 여기서 함수는 끝난다.
    # else:
    #     return b
    # print('hahahah')

    # if a >= b:
    #     return a
    # return b  #b가 더 큰값이길 바란다.

    if a >= b:
        b = a   #b가 크면 그대로 출력, a가 크면 b에 a를 덮어서 사용
    return b

print(max2(10, 24))

# 4개의 정수 중에서 큰 숫자를 찾느 힘수르 만드세요.

def max4(a, b, c, d):
    return max2(max2(a, b), max2(c, d))

print(max4(1,2,3,4))
print(max4(2,3,4,1))
print(max4(3,4,1,2))
print(max4(4,1,2,3))

# 한국시리즈
def max4(a, b, c, d):
    return max2(max2(max2(a, b),c), d)

print(max4(1,2,3,4))
print(max4(2,3,4,1))
print(max4(3,4,1,2))
print(max4(4,1,2,3))