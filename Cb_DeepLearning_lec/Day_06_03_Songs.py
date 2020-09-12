# Day_06_03_Songs.py
import requests
import re

# HTTP
# GET
# https://www.google.com/search?q=python&oq=python
# POST


def show_songs(code, page):
    payload = {
        'S_PAGENUMBER': page,       # 1
        'S_MB_CD': code,            # 'W0726200'
        # 'S_HNAB_GBN': 'I',
        # 'hanmb_nm': '권지용',
        # 'sort_field': 'SORT_PBCTN_DAY',
    }

    url = 'https://www.komca.or.kr/srch2/srch_01_popup_mem_right.jsp'
    received = requests.post(url, data=payload)
    # print(received)
    # print(received.text)

    # 문제
    # 지드래곤의 저작권 데이터를 깨끗하게 출력하세요
    tbody = re.findall(r'<tbody>(.+?)</tbody>',
                       received.text,
                       re.DOTALL)
    # print(tbody)
    # print(len(tbody))
    # print(tbody[1])

    # 문제
    # 노래를 감싸고 있는 무언가를 찾으세요
    trs = re.findall(r'<tr>(.+?)</tr>', tbody[1], re.DOTALL)
    # print(trs)
    # print(len(trs))

    # 문제
    # 제목, 가수 등등의 데이터를 찾아보세요
    for tr in trs:
        # tr = re.sub(r' <img src="/images/common/control.gif"  alt="" />',
        #             '', tr)
        # tr = re.sub(r' <img src="/images/common/control.gif" alt="" />',
        #             '', tr)
        tr = re.sub(r' <img .+? />', '', tr)
        tr = re.sub(r'<br/>', ',', tr)
        tds = re.findall(r'<td>(.+)</td>', tr)
        tds[0] = tds[0].strip()
        print(tds)

    return len(trs) > 0


# for page in range(18):
#     show_songs('W0726200', page)

# show_songs('W0726200', 1000)

page = 1
while show_songs('W0726200', page):
    print('----------------', page)
    page += 1
