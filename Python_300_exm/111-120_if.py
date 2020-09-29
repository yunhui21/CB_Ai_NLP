# 111-120_if.py

# 111
# data = input('입력하세요:')
# print('111:', data*2)

# 112
# num = int(input('숫자를 입력하세요:'))
# print('112:', num+10)

# 113
# num = input('')
# if  int(num) % 2 ==0:
#     print('113:','짝수')
# else:
#     print('113:','홀수')

# 114
# nums = int(input('')) + 20
# if nums <= 255:
#     print('114:', nums)
# else:
#     print('114:', 255)

# 115
# num = int(input(''))-20
# if num < 0:
#     print('115:', 0)
# elif num > 255:
#     print('115:', 255)
# else:
#     print('115:', num)

# 116
# time = input('현재시각:')
# if time[-2:] == "00":
#     print('116:', "정각입니다.")
# else:
#     print('116:', "정각이 아닙니다.")

# 117
# fruit = ['사과', '포도', '홍시']
# quest = input('좋아하는 과일은?')
# if quest in fruit:
#     print('117:', '정답입니다.')
# else:
#     print('117:', '정답이아닙니다.')

# 118
# warn_investment_list = ['Microsoft', 'Google', 'Naver','Kakao', 'SAMSUNG', 'LG']
# quest = input('종목은?')
# if quest in warn_investment_list:
#     print('118:','투자 경고 종목입니다')
# else:
#     print('118:', '투자 경고 종목이 아닙니다.')

# 119
# fruit = {'봄':'딸기', '여름':'토마토', '가을':'사과'}
# weather = input('제가 좋아하는 계절은:')
# if weather in fruit.keys():
#     print('119:', '정답입니다.')
# else:
#     print('119:', '오답입니다.')

# 120
fruit = {'봄':'딸기', '여름':'토마토', '가을':'사과'}
choice = input('좋아하는 과일은?')
if choice in fruit.values():
    print('120:', '정답입니다.')
else:
    print('120:', '오답입니다.')