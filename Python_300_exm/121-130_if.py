# 121-130_if.py

# 121
# alpha = input("")
# if alpha.islower():
#     print('121:', alpha.upper())
# else:
#     print('121:', alpha.lower())

# 122
# score = int(input('score:'))
# if 81<= score <= 100:
#     print('122: Grade is A')
# elif 61 <= score <= 80:
#     print('122: Grade is B')
# elif 41 <= score <= 60:
#     print('122: Grade is C')
# elif 21 <= score <= 40:
#     print('122: Grade id D')
# else:
#     print('122: Grade is E')

# 123
# change = {'달러': 1167,
#           '엔' : 1.096,
#           '유로': 1268,
#           '위안': 171}
# won = input('입력:')
# num, currency = won.split()
# print('123:', float(num)*change[currency], '원')

# 124
# num1 = input('number1:')
# num2 = input('number2:')
# num3 = input('number3:')
# num1 = int(num1)
# num2 = int(num2)
# num3 = int(num3)
#
# if num1 > num2 and num1 > num3:
#     print('124:', num1)
# elif num2 > num1 and num2 > num3:
#     print('124:', num2)
# else:
#     print('124:', num3)

# 125
# your_phone = input("휴대전화 번호 입력:")
# num= your_phone.split('-')[0]
# if num == '011':
#     com = 'SKT'
# elif num == '016':
#     com = 'KT'
# elif num == '019':
#     com = 'LGU'
# else:
#     com = '알수없음'
# print(f'당신은 {com} 사용자입니다.')

# 126
# post = input('우편번호:')
# post = post[:3]
# if post in ['010', '011', '012']:
#     print('126:', '강복구')
# elif post in ['013', '014', '015']:
#     print('126:', '도봉구')
# else:
#     print('126:', '노원구')

# 127
# identi = input('주민등록번호:')
# identi = identi[7]
# if identi in ['1','3']:
#     print('127:', '남자')
# else:
#     print('127:', '여자')

# identi = input('주민등록번호:')
# identi = identi.split('-')[1]
# if identi[0] == '1' or identi[0] == '3':
#     print('127:', '남자')
# else:
#     print('127:', '여자')

# 128
# iden = input('주민등록번호:')
# local = iden.split('-')[1]
# if 0 <= int(local[1:3]) <= 8:
#     print('128:', '서울입니다.')
# else:
#     print('128:', '서울이 아닙니다.')

# 129
identi = input('주민등록번호:')
number = identi.split('-')
first = number[0]*2 + number[1]*3 + number[2]*4 + number[3]*5 + \
        number[4]*6 + number[5]*7 + number[7]*8 + number[8]*9 + \
        number[9]*2 + number[10]*3 + number[11]*4 + number[12]*5
second = first % 11
third = second-7
if third == 4:
    print('129: 유효한 주민등록번호업니다.')
else:
    print('129: 유효하지 않은 주민등록번호입니다.')