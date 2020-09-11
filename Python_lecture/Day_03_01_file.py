# Day_03_01_file.py
# 폴더 : Data
import pandas as pd

# def read_1(filename):
#     f = open('filename', 'r', encoding='utf-8')
#     f.close()



# 경로(path) 상대경로, 절대경로
# filename=./data/poem.txt
# filename=../CB_Ai_NLP/data/poem.txt
# filename = '../data/poem.txt'
# read_1(filename)
#
f = open('../data/poem.txt', 'r', encoding='utf-8')
lines = f.readlines()
for line in lines:
    # print(line, end='')
    print(line.strip())
    # strip:문자열 양쪽끈에 있는 공백을 제거
f.close