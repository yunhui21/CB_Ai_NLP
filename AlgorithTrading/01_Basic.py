# 01_Basic.py
print('helloWorld')
print('안녕하세요')
print('-'*50)
#------------------#
print('"주식은 대박이다"')
print(6010000*10)
print(601000 * 0.97)
print(601000 - 587000)
print(180 * 10 + 10000)
print((180 * 10)+ 10000)

print('-'*50)

naver = 601000
print(naver)
print(naver*10)

# 월요일 종가 = 10,000 - (10,000 * 0.3) = 7,000
# 화요일 종가 = 7,000 - (7,000 * 0.3) = 4,900

monday_end_price = 10000 - (10000 * 0.3)
tuesday_end_price = monday_end_price - (monday_end_price * 0.3)
print(tuesday_end_price)

x = 100
id(x)
print(id(x))
y = 100
id(y)
print(id(y))

x = 10000
y = 10000
print(id(x))
print(id(y))

print('-'*50)

mystring = 'hello world'
mystring1 = 'a'
mystring2 = "a"
mystring3 = 'abc mart'

print(mystring)         # hello world
print(len(mystring))    # 11
print(mystring[:5])     # hello
print(mystring[6:])     # world
print(mystring[6:-1])   # worl

my_jusik = 'naver daum'
print(my_jusik)     # naver daum
print(my_jusik[:5]) # naver

print(my_jusik.split(' '))  #['naver', 'daum']
print(my_jusik.split(' ')[0]) # naver

split_jusik = my_jusik.split(' ')
print(split_jusik)      # ['naver', 'daum']
print(split_jusik[0])   # naver


daum = 'daum'
kakao = 'kakao'

print(daum + kakao)       # daumkakao
print(daum + ' ' + kakao) # daum kakao

daum_kakao =  daum + kakao
print(daum_kakao)   # daumkakao
print('-'*50)
#-------------------------------------#

print(type(7000))       # <class 'int'>
print(type(3.141592))   # <class 'float'>'
print(type('python'))   # <class 'str'>

daum_juga = 89000
naver_juga = 751000
total = daum_juga*100 + naver_juga*20
print(total)






