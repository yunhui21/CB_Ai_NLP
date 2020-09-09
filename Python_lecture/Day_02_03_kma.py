# Day_02_03_kma.py

import requests
import re

url = 'https://search.naver.com/search.naver?query=%EB%82%A0%EC%94%A8&ie=utf8&sm=whl_nht'
recvd = requests.get(url)
print(recvd)
print(recvd.text)