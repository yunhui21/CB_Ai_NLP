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
기준가격 = 9980
1차 계산: 기준 가격에 0.3을 곱한다.
9980 x 0.3 = 2994원
2차 계산: 기준 가격에 호가 가격단위에 미만을 절사한다. 
2,990원(기준 가격(9,980원)의 호가 가격단위인 10원 미만 절사)
3차 계산 : 기준 가격에 2차 계산에 의한 수치를 가감하되, 해당 
가격의 호가 가격단위 미만을 절사한다.
합산가격 : 9,980원 + 2,990원 = 12,970원
호가 가격 단위 적용: 12,970원의 호가 가격단위인 50원 미만 절사(2차절사)
상한가 : 12,950원'''

cur_price = 9980
print(cur_price >= 5000 and cur_price < 10000)

day1 = 10000
day2 = 13000
print(((day2-day1)==(day1*0.3)) or ((day2-day1)> (day1*0.292)))


# 3)파이썬 if 문
wikibooks_cur_price = 8000
if wikibooks_cur_price < 9000:
    print('sell 10')

wikibooks_cur_price = 11000
if wikibooks_cur_price >= 10000:
    print('buy 5')
    print('buy 5')
    print('buy 5')

# 3)-1 if ~ else 문
wikibooks_cur_price = 11000
if wikibooks_cur_price >= 10000:
    print('buy 10')
else:
    print('holding')

wikibooks_cur_price = 9000
if wikibooks_cur_price >= 10000:
    print('buy 10')
else:
    print('holding')

# 3)-2 if ~ elif ~ else 문
price = 7000
if price <= 1000:
    print('bid = 1')
elif 1000 < price <= 5000:
    print('bid = 5')
elif 5000 < price <= 10000:
    print('bid = 10')
elif 10000 < price <= 50000:
    print('bid = 50')
elif 50000 < price <= 100000:
    print('bid = 100')
elif 100000 < price <= 500000:
    print('bid = 500')
else:
    print('bid 1000')

# 4) for문
for i in [0,1,2,3,4,5,6,7,8,10]:
    print(i)
# 4)-1 for range
for i in range(1,10):
    print(i)

print(list(range(1,10,2)))

for i in range(0,11):
    print(i)

for i in list(range(0,11)):
    print(i)

# 4)-2 for와 리스트
stocks = ['Naver', 'samsung', 'SK Hynix']
for company in stocks:
    print('%s: Buy 10'%company)

# 4)-3 for와 튜플
stocks = ('naver', 'samsung', 'sk hynix')
for company in stocks:
    print('%s: Buy 10'%company)



