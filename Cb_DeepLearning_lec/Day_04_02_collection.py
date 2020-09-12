# Day_04_02_collection.py

a = [1, 3, 7, 9]

for i in a:
    print(i, end=' ')
print()

# 문제
# 리스트를 거꾸로 출력하세요
for i in reversed(a):
    print(i, end=' ')
print()

# 튜플 : 상수 버전의 리스트 (읽기 전용 리스트)
b = (1, 3, 7, 9)
print(b)
print(b[0], b[1])

for i in b:
    print(i, end=' ')
print()

# b[0] = -1         # error
# b.append(99)      # error

c1 = (1, 4)
c2 = 1, 4           # packing
print(c1)
print(c2)

c3, c4 = 1, 4
c3, c4 = c2         # unpacking
# c3, c4, c5 = c2   # error
print(c3, c4)


def multi(d1, d2):
    return d1 + d2, d1 * d2


e = multi(3, 5)
print(e, e[0], e[1])

e1, e2 = multi(3, 5)
print(e1, e2)
print('-' * 30)

# set (집합)
g = {1, 3, 5, 1, 3, 5, 1, 3, 5, }   # 순서 없음
print(g)

h = [1, 3, 5, 1, 3, 5, 1, 3, 5, ]   # 순서 보장
print(h)
print(set(h))
print(list(set(h)))

for i in g:
    print(i)

print('-' * 30)

# 딕셔너리 (사전)
# 영한사전 : 영어 단어를 찾으면 한글 설명 나옴
# 영어단어 : key
# 한글설명 : value

info = {'age': 21, 'addr': 'ochang', 'hobby': 'minton', 12: 34}
print(type(info), type(set()), type((3,)))
print(info[12])
# <class 'dict'> <class 'set'> <class 'tuple'>
info = dict(age=21, addr='ochang')
print(info)
print(info['age'], info['addr'], info['hobby'])

info.pop('hobby')
info.pop('addr')

info['blood'] = 'AB'        # insert
info['blood'] = 'O'         # update

# info.popitem()              # 마지막에 추가한 항목 삭제
print('-' * 30)

print(info.keys())
print(info.values())
print(info.items())

for k in info.keys():
    print(k, info[k])

# 문제
# items()를 반복문에 적용하세요
p = list(info.items())
print(p)

for i in info.items():
    print(i, i[0], i[1])

for k, v in info.items():
    print(k, v)

# range, reversed, enumerate
a = ['A', 'B', 'C']
for i in a:
    print(i)

# 문제
# 아래 코드를 파이썬답게 바꾸세요
for i in enumerate(a):
    print(i)

for i, letter in enumerate(a):
    print(i, letter)

# 문제
# items()에 enumerate를 연결하세요
for i in enumerate(info.items()):
    # print(i, i[0], i[1], i[1][0], i[1][1])
    print(i[0], i[1][0], i[1][1])

for i, kv in enumerate(info.items()):
    # print(i, kv, kv[0], kv[1])
    print(i, kv[0], kv[1])

for i, (k, v) in enumerate(info.items()):
    print(i, k, v)

for k in info:
    print(k, info[k])


print('\n\n\n')