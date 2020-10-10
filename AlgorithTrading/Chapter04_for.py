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

# 4)-4 for와 딕셔너리
stocks = {'Naver':10, 'Samsung':5, 'SK Hynix':30}
for company, stock_num in stocks.items():
    print('%s : Buy %s' %(company, stock_num))

for company in stocks.keys():
    print('%s : Buy %s' % (company, stocks[company]))

# 4)- 5 while문
i = 0
while i <= 10:
    print(i)
    i = i + 1
print(i)

# 4)-5-1 while문을 이용한 상한가 계산
wikibooks_cur_price = 10000
#first price
first_price = wikibooks_cur_price + wikibooks_cur_price*0.3
print(first_price)
#second price
second_price = first_price + first_price * 0.3
print(second_price)
#third_price
third_price = second_price + second_price * 0.3
print(third_price)

wikibooks = 10000
day = 1
while day < 6:
    wikibooks = wikibooks + wikibooks * 0.3
    day = day + 1
    print(wikibooks)
# while 문의 조건 확인
# 조건을 만족하면 while 문 내부의 코드를 차례로 수행하고, 조건을 만족하지 않으면 while 문의 다음 문장을 수행
# 위의 과정에서 while문 내부의 코드를 차례로 수행한 경우 다시 while 문의 조건 확인으로 이동

# 4)-5-2 while과 if
num = 0
while num <= 10:
    if num % 2 == 1:
        print(num)
    num +=1

# 4)-5-3 break와 continue
while 1:
    print('find stocks')
    break
num = 0
while 1:
    print(num)
    if num == 10:
        break
    num += 1

num = 0
while num < 10:
    num += 1
    if num == 5:
        continue
    print(num)

# 4)-5-6 중첩루프
for i in  [1,2,3,4]:
    for j in [1,2,3,4]:
        pass

'''
1층에 가서 1층의 각 세대에 신문 배달
2층에 가서 2층의 각 세대에 신문 배달
3층에 가서 3층의 각 세대에 신문 배달
4층에 가서 4층의 각 세대에 신문 배달
'''
for i in [101, 102, 103, 104 ]:
    print('Delivery :', i)
apart = [[101, 102, 103, 104],
        [201, 202, 203, 204],
        [301, 302, 303, 304],
        [401, 402, 403, 404]]
for i in apart:
    print('Delivery :', i)

for floor in apart:
    for room in floor:
        print('Newspaper delivery:', room)