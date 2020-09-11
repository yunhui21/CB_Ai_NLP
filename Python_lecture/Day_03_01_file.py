# Day_03_01_file.py
# 폴더 : Data
import pandas as pd

# def read_1(filename):
#     f = open('filename', 'r', encoding='utf-8')
#     line = f.readlines()
#     print(line)
#     for line in lines:
#         # print(line, end='')
#         print(line.strip())
#         # strip:문자열 양쪽끝에 있는 공백을 제거
#     f.close()
# # 경로(path) 상대경로, 절대경로
# # filename=./data/poem.txt
# # filename=../CB_Ai_NLP/data/poem.txt
# filename = '../data/poem.txt'
# read_1(filename)
#
f = open('../data/poem.txt', 'r', encoding='utf-8')
lines = f.readlines()#한번에 다 읽기
print(lines)
for line in lines:
    # print(line, end='')
    print(line.strip())
    # strip:문자열 양쪽끝에 있는 공백을 제거
f.close
print('-'*50)


temp = '\t\t\n\n apple koong \n\n\t\t'
print('[{}]'.format(temp))
temp = temp.strip()
print('[{}]'.format(temp))
print('-'*50)


f = open('../data/poem.txt', 'r', encoding='utf-8')
while True:
    line = f.readline()#한줄씩 읽기
    # if len(line) == 0:
    if not line:
        break
    print(line.strip())
f.close()
print('-'*50)

f = open('../data/poem.txt', 'r', encoding='utf-8')

for line in f:
    print(line.strip())

f.close()


print('-'*50)

with open('../data/poem.txt', 'r', encoding='utf-8') as f:
    for line in f:
        print(line.strip())
    # f.close() with는 무조건 close를 호출한다.
