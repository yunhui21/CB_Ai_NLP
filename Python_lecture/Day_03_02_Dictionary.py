# Day_03_02_Dictionary.py

a = {}              # dict
print(type(a), a)

a = {1}             # set
print(type(a), a)

a = {1 : 2}         # dict
print(type(a), a)


b = {1,3,5,1,3,5,1,3,5}
print(b)            # 중복된 데이터를 허용하지 않는다. 모두 정렬된 값을 반돤한다.
print(type(b))

c = [1,3,5,1,3,5,1,3,5]
print(c)            # 중복된 데이터를 허용한다.
print(type(c))

#문제
#리스트에 들어있는 중복데이터를 모두 제거해주세요.
c = list(set(c))
print(c)

d = {'color': 'red', 'price' : 100}
d = dict(color='red', price=1000)       #순서가 중요하지 않는다.. 값을 빨리 찾는게 중요
print(d)
print(type(d))
print(d['color'], d['price'])

