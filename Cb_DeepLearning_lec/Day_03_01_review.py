# Day_03_01_review.py

# 문제
# 1에서부터 5까지 출력하세요
# for문으로 만들어 보세요

for i in range(5):
    print(i + 1, end=' ')
print()

for i in range(1, 6):
    print(i, end=' ')
print()

i = 1
while i < 6:
    print(i, end=' ')
    i += 1
print()
print('-' * 30)

# 문제
# 이름을 거꾸로 출력하세요
name = 'kim'  #input('name : ')
print(name)
print(type(name))       # <class 'str'>

name_len = len(name)
print(name_len)

print(name[0], name[1], name[2])
# print(name[3])

for i in range(name_len):
    print(name[i])

print('-' * 30)










