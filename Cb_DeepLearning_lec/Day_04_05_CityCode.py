# Day_04_05_CityCode.py
import requests
import re

# 문제
# 기상청 지역 코드를 읽어와서
# json에서와 동일하게 정규 표현식을 사용해서 출력하세요
url = 'http://www.kma.go.kr/DFSROOT/POINT/DATA/top.json.txt'
received = requests.get(url)
print(type(received.content))   # <class 'bytes'>

text = received.content.decode('utf-8')
print(text)

# [{"code":"11","value":"서울특별시"},,{"code":"50","value":"제주특별자치도"}]

# codes = re.findall(r'[0-9]+', text)
# codes = re.findall(r'[0-9][0-9]', text)
# codes = re.findall(r'"code":".+?"', text)
codes = re.findall(r'"code":"(.+?)"', text)
print(codes)

# values = re.findall(r'[가-힣]+', text)
values = re.findall(r'"value":"(.+?)"', text)
print(values)

# [{"code":"11","value":"서울특별시"},{"code":"26","value":"부산광역시"}]
#                                ^   ^    ^ ^  ^ ^     ^ ^        ^
