#Quiz_03

# 문제 3-1
naver_closing_price = [488500, 500500, 501000, 461500, 474500]
print('naver_closing_price =',naver_closing_price)

# 문제 3-2
# print(max(naver_closing_price))
Max = max(naver_closing_price)
print('가장 높은 종가:',Max)


# 문제 3-3
# print(min(naver_closing_price))
Min = min(naver_closing_price)
print('가장낮은 종가:',Min)

# 문제 3-4
print('가격차:',Max - Min)

# 문제 3-5
print('수요일종가:', naver_closing_price[2])

# 문제 3-6
naver_closing_price2 = {'09/11':488500, '09/10':500500, '09/09':501000, '09/08':461500, '09/07':474500}
print(naver_closing_price2)

# 문제3-7
print('09/09일 종가:',naver_closing_price2['09/09'])