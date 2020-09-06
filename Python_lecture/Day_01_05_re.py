# Day_01_05_re.py
import re

db = '''3412    Bob 123
3834  Jonny 333
1248   Kate 634
1423   Tony 567
2567  Peter 435
3567  Alice 535
1548  Kerry 534'''

db = '3412    Bob 123\n'\
     '3834  Jonny 333\n'\



# print(db)
# ''' '''
# raw 문자열 생성
# 파이썬 문자열: \n, \t, \d

# ns = re.findall(r'[0-9]', db)    #findall 모두 찾아줘. 패턴을 찾아줘.
# print(ns)   #['3', '4', '1', '2', '1', '2',


ns = re.findall(r'[0-9]+', db)    #findall 모두 찾아줘. 패턴을 찾아줘.
print(ns)   #['3412', '123', '3834', '333',

# 문제
# 이름만 찾아보세요.

# name = re.findall(r'[A-z]+', db)
name = re.findall(r'[A-Za-z]+', db)
name = re.findall(r'[A-Z][a-z]+', db)   #+가 붙은 부분외에 대문자의 한글자를 붙여라.

print(name)

# 문제
# T로 시작하는 이름만 찾아보세요.
T_name = re.findall(r'[T][a-z]+', db)   #+가 붙은 부분외에 대문자의 한글자를 붙여라.
print(T_name)

# T로 시작하지 않는 이름만 찾아보세요.
T_name = re.findall(r'[^T][a-z]+', db)   #+가 붙은 부분외에 대문자의 한글자를 붙여라.
print(T_name)   #['Bob', 'Jonny', 'Kate', 'ony', 'Peter', 'Alice', 'Kerry']

print(re.findall(r'[A-SU-Z][a-z]+', db))



