# Day_04_06_weather.py
import requests
import re

# f = open('data/weather.csv', 'w', encoding='utf-8')

url = 'http://www.kma.go.kr/weather/forecast/mid-term-rss3.jsp?stnId=108'
received = requests.get(url)
# print(received.text)

# temp = re.findall(r'<province>(.+?)</province>',
#                   received.text)
# print(temp)
# print(len(temp))

# .+ : 탐욕적(greedy)
# .+? : 비탐욕적(non-greedy)
# re.DOTALL : 개행 문자 무시.
#    찾고자 하는 것이 여러 줄에 걸쳐 있을 때.
locations = re.findall(r'<location wl_ver="3">(.+?)</location>',
                       received.text, re.DOTALL)
# print(len(locations))
# print(locations[0])

for loc in locations:
    # 문제
    # province와 city를 같이 찾아보세요
    # prov = re.findall(r'<province>(.+?)</province>', loc)
    # city = re.findall(r'<city>(.+?)</city>', loc)
    # print(prov[0], city[0])

    # pattern = r'<province>(.+?)</province>\r\n\t\t\t\t<city>(.+?)</city>'
    pattern = r'<province>(.+?)</province>.+?<city>(.+?)</city>'
    prov_city = re.findall(pattern, loc, re.DOTALL)
    # print(prov_city[0])
    prov, city = prov_city[0]

    # 문제
    # data를 찾아보세요
    data = re.findall(r'<data>(.+?)</data>',
                      loc, re.DOTALL)
    # print(len(data))

    # 문제
    # mode, tmEf, wf, tmn, tmx, rnSt 데이터를 출력하세요
    for datum in data:
        # 문제
        # mode, tmEf, wf, tmn, tmx, rnSt 데이터를 한번에 찾아보세요
        # mode = re.findall(r'<mode>(.+)</mode>', datum)
        # tmEf = re.findall(r'<tmEf>(.+)</tmEf>', datum)
        # wf = re.findall(r'<wf>(.+)</wf>', datum)
        # tmn = re.findall(r'<tmn>(.+)</tmn>', datum)
        # tmx = re.findall(r'<tmx>(.+)</tmx>', datum)
        # rnSt = re.findall(r'<rnSt>(.+)</rnSt>', datum)
        #
        # print(prov_city[0], mode[0], tmEf[0], wf[0], tmn[0], tmx[0], rnSt[0])

        items = re.findall(r'<mode>(.+)</mode>.+?<tmEf>(.+)</tmEf>',
                           datum, re.DOTALL)
        # print(prov_city[0], items[0])
        print(*prov_city[0], *items[0])

        items = re.findall(r'<.+?>(.+?)</.+?>', datum)
        mode, tmEf, wf, tmn, tmx, rnSt = items
        # print(prov, city, mode, tmEf, wf, tmn, tmx, rnSt)
        # print(prov, city, mode, tmEf, wf, tmn, tmx, rnSt,
        #       file=f, sep=',')

        # base = '{},{},{},{},{},{},{},{}\n'
        # row = base.format(prov, city, mode, tmEf, wf, tmn, tmx, rnSt)
        # f.write(row)

# f.close()

# 제주도 서귀포
# A02 2020-08-11 00:00 흐림 26 29 40
# A02 2020-08-11 12:00 흐림 26 29 40
# ==>
# 제주도 서귀포 A02 2020-08-11 00:00 흐림 26 29 40
# 제주도 서귀포 A02 2020-08-11 12:00 흐림 26 29 40

# 문제
# 기상청 데이터를 weather.csv 파일에 저장하세요



