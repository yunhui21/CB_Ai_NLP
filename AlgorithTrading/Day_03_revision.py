# Day_03_revision.py
interest1 = '삼성전자'
interest2 = 'LG전자'
interest3 = '네이버'

print(interest1)

interest = ['삼성전자', 'LG전자', '네이버']
print(interest)
print(interest[0], interest[1], interest[2])

daishin = [9130, 9150, 9300, 9400]
mystock = ['naver', 5000]
yostock = [] # 빈리스트 생성

my_list = ['naver', 5000]
print(my_list[0], my_list[1])

kospi = ['삼성전자', 'SK하이닉스', '현대차', '한국전력', '아모레퍼시픽', '제일모직', '삼성전자우','삼성생명','naver', '현대모비스']
print("시가총액 5위:",kospi[4])

kospi_top5 = kospi[:5]
print(kospi_top5)

kospi.append('sk텔레콤')
print(kospi)
kospi.insert(2,'daum')
print(kospi)
print(len(kospi))
del kospi[2]
print(kospi)
print(len(kospi))

# list: append, inster, del

# 2)튜플
# list = [], tuple = ()
# list =원소변경 가능, tuple = 원소변경 불가

# 3)딕셔너리
cur_price = {}
print(type(cur_price))  # <class 'dict'>
cur_price['daeshin']=10000
print(cur_price)        # {'daeshin': 10000}
cur_price['Kakao'] = 5000
print(cur_price)        # {'daeshin': 10000, 'Kakao': 5000}
cur_price['daum'] =50000
print(cur_price)        # {'daeshin': 10000, 'Kakao': 5000, 'daum': 50000}
print(len(cur_price))
print(cur_price['daeshin'])
cur_price['Naver']= 80000
print(cur_price)li
del cur_price['daeshin']
print(cur_price)


