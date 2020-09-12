# Day_04_03_json.py
import json
import requests

a = '{"ip": "8.8.8.8"}'
print(a)
print(type(a))

b = json.loads(a)
print(b)
print(type(b))

# 문제
# 딕셔너리를 문자열로 변환하세요
c = json.dumps(b)
print(c)
print(type(c))

# 문제
# 아래 데이터에서 value에 해당하는 데이터만 출력하세요
dt = '''{
   "time": "03:53:25 AM",
   "milliseconds_since_epoch": 1362196405309,
   "date": "03-02-2013"
}'''

dt2 = json.loads(dt)
print(dt2)
print(dt2['time'], dt2['milliseconds_since_epoch'], dt2['date'])

for k in dt2:
    print(dt2[k])

print(dt2.values())
print('-' * 30)

# 문제
# 기상청 지역 코드를 읽어와서 도시 이름과 코드만 깨끗하게 출력하세요
# url = 'http://www.kma.go.kr/DFSROOT/POINT/DATA/top.json.txt'
# received = requests.get(url)
# print(received)
# print(received.text)
# print(received.content)
# print(type(received.content))   # <class 'bytes'>
#
# text = received.content.decode('utf-8')
# print(text)

f = open('data/code.txt', 'r', encoding='utf-8')
text = f.read()
f.close()

print(type(text))

items = json.loads(text)
print(items)

for item in items:
    # print(item)
    print(item['code'], item['value'])


print('\n\n\n')