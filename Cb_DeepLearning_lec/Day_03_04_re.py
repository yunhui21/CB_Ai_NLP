# Day_03_04_re.py
import re

db = '''3412    [Bob] 123
3834  Jonny 333
1248   Kate 634
1423   Tony 567
2567  Peter 435
3567  Alice 535
1548  Kerry 534'''

# print(db)

number = re.findall(r'[0-9]', db)
print(number)

# 1, 15, 154, 1548
numbers = re.findall(r'[0-9]+', db)
print(numbers)

# 문제
# 이름만 찾아보세요
print(re.findall(r'[A-z]+', db))       # wrong
print(re.findall(r'[A-Za-z]+', db))    # just
print(re.findall(r'[A-Z][a-z]+', db))  # good

# 문제
# 1. T로 시작하는 이름만 찾아보세요
# 2. T로 시작하지 않는 이름만 찾아보세요
print(re.findall(r'T[a-z]+', db))
print(re.findall(r'[T][a-z]+', db))
print(re.findall(r'[^T][a-z]+', db))
print(re.findall(r'[ABCDEFGHIJKLMNOPQRSUVWXYZ][a-z]+', db))
print(re.findall(r'[A-SU-Z][a-z]+', db))




