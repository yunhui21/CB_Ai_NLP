# Day_08_02_weather.py
import requests
import re

f = open('data/weather.csv', 'w', encoding='utf-8')

url = 'http://www.kma.go.kr/weather/forecast/mid-term-rss3.jsp?stnId=108'
received = requests.get(url)
# print(received)
# print(received.text)

# 문제
# province를 찾으세요
# temp = re.findall(r'<province>(.+)</province>',
#                   received.text)
# print(temp)
# print(len(temp))

# 문제
# location을 찾으세요

# DOTALL : 개행문자 무시. 찾으려고 하는 것이 여러 줄에 있을 때
# .+ : 탐욕적(greedy)
# .+? : 비탐욕적(non-greedy)
locations = re.findall(r'<location wl_ver="3">(.+?)</location>',
                       received.text, re.DOTALL)
# print(len(locations))
# print(locations[:2])

# 문제
# province와 city를 찾으세요
for loc in locations:
    # print(loc)
    # prov = re.findall(r'<province>(.+)</province>', loc)
    # city = re.findall(r'<city>(.+)</city>', loc)
    # print(prov[0], city[0])

    # 문제
    # province와 city를 한번에 찾으세요
    # pattern = r'<province>(.+)</province>\r\n\t\t\t\t<city>(.+)</city>'
    pattern = r'<province>(.+)</province>.+<city>(.+)</city>'
    prov_city = re.findall(pattern, loc, re.DOTALL)
    prov, city = prov_city[0]
    # print(prov, city)

    # 문제
    # data를 찾으세요
    data = re.findall(r'<data>(.+?)</data>', loc, re.DOTALL)
    # print(len(data))

    for datum in data:
        # print(datum)
        # mode = re.findall(r'<mode>(.+?)</mode>', datum)
        # tmEf = re.findall(r'<tmEf>(.+?)</tmEf>', datum)
        # wf = re.findall(r'<wf>(.+?)</wf>', datum)
        # tmn = re.findall(r'<tmn>(.+?)</tmn>', datum)
        # tmx = re.findall(r'<tmx>(.+?)</tmx>', datum)
        # rnSt = re.findall(r'<rnSt>(.+?)</rnSt>', datum)
        # print(prov, city, mode[0], tmEf[0], wf[0], tmn[0], tmx[0], rnSt[0])

        # 문제
        # datum 안에 들어있는 것들을 한번에 찾으세요
        # sub_pattern = r'<mode>(.+?)</mode>.+<tmEf>(.+?)</tmEf>.+<wf>(.+?)</wf>.+<tmn>(.+?)</tmn>.+<tmx>(.+?)</tmx>.+<rnSt>(.+?)</rnSt>'
        # items = re.findall(sub_pattern, datum, re.DOTALL)
        # print(items[0])
        # mode, tmEf, wf, tmn, tmx, rnSt = items[0]
        # print(prov, city, mode, tmEf, wf, tmn, tmx, rnSt)

        sub_pattern = r'<.+>(.+?)</.+>'
        items = re.findall(sub_pattern, datum)
        # print(items)
        mode, tmEf, wf, tmn, tmx, rnSt = items
        # print(prov, city, mode, tmEf, wf, tmn, tmx, rnSt)
        print(prov, city, *items)

        # print(prov, city, mode, tmEf, wf, tmn, tmx, rnSt,
        #       file=f, sep=',')

        # line = '{},{},{},{},{},{},{},{}\n'.format(prov, city, mode, tmEf, wf, tmn, tmx, rnSt)
        # f.write(line)

        f.write(prov + ',')
        f.write(city + ',')
        f.write(mode + ',')
        f.write(tmEf + ',')
        f.write(wf + ',')
        f.write(tmn + ',')
        f.write(tmx + ',')
        f.write(rnSt + '\n')

        # 210.125.150.125

f.close()

# 문제
# 파싱한 결과를 weather.csv 파일에 저장하세요

# 제주도 서귀포
# A02 2020-07-19 00:00 흐리고 비 24 28 80
# A02 2020-07-19 12:00 흐리고 비 24 28 60
# ==>
# 제주도 서귀포 A02 2020-07-19 00:00 흐리고 비 24 28 80
# 제주도 서귀포 A02 2020-07-19 12:00 흐리고 비 24 28 60
