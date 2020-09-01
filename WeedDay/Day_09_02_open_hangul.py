# Day_09_02_open_hangul.py
import requests
import re

# 문제
# 오픈한글에 있는 API를 사용해서 한글 단어에 일치하는 영문 자판을 출력하세요

def kor2eng(kor):
    url = 'https://openhangul.com/nlp_ko2en?q=' + kor
    received = requests.get(url)
    # print(received.text)

    pattern = r'<img src="images/cursor.gif"><br>(.+)</pre>'
    eng = re.findall(pattern, received.text, re.DOTALL)
    return eng[0].strip()


print(kor2eng('여름바다'))      # dufmaqkek
print(kor2eng('싹쓰리'))       # TkrTmfl
print(kor2eng('깡'))          # Rkd

# 210.125.150.125


