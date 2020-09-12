# Day_02_01_if.py

# if 비가 오는 상황이라면:
# print('비가 온다')
# print('비가 오지 않는다')

a = 3

#      0 1 2 3 4 5
# %2 : 0 1 0 1 0 1

if a % 2 == 1:
    print('홀수')
else:
    print('짝수')

if a % 2:
    print('홀수')
else:
    print('짝수')

if a:
    print('홀수')
else:
    print('짝수')

# 문제
# 입력 받은 정수가 음수인지 양수인지 0인지 알려주세요
a = 10

if a > 0:
    print('양수')
else:
    # print('양수가 아니라면')
    # print('음수 or 제로')
    if a < 0:
        print('음수')
    else:
        print('제로')

print('end')

a = -10

if a > 0:
    print('양수')
elif a < 0:
    print('음수')
else:
    print('제로')

print('end')

# 문제
# 성적을 학점으로 변환하세요
# 성적 : 0 ~ 100
# 학점 : A(90) B(80) C(70) D(60) E(나머지)
score = 95
# score = 85
# score = 75
# score = 65
# score = 55

# if 90 <= score <= 100:
#     grade = 'A'
# if 80 <= score <= 89:
#     grade = 'B'
# if 70 <= score <= 79:
#     grade = 'C'
# if 60 <= score <= 69:
#     grade = 'D'
# if 0 <= score <= 59:
#     grade = 'F'

# if 90 <= score <= 100:
#     grade = 'A'
# elif 80 <= score <= 89:
#     grade = 'B'
# elif 70 <= score <= 79:
#     grade = 'C'
# elif 60 <= score <= 69:
#     grade = 'D'
# elif 0 <= score <= 59:
#     grade = 'F'

# score = 150

if score < 0 or score > 100:
    exit(-1)

if 90 <= score:
    grade = 'A'
elif 80 <= score:
    grade = 'B'
elif 70 <= score:
    grade = 'C'
elif 60 <= score:
    grade = 'D'
else:
    grade = 'F'

print(grade)






# 210.125.150.125
