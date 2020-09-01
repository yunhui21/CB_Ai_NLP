#DAy_02_01_comprehension.py

text = 'John works at Intel'

#문제
#문자열로부터 각 단어의 길이로 구성된 리스트를 만드세요
#[4,5,2,5]
#[i for i in text]

print([i for i in text])
#split 함수 사용
print([i for i in text.split()])
#단어의 길이를 변환
print([len(i) for i in text.split()])
#[]를 {}-dit,set (dit:key, value), set(
print({len(i) for i in text.split()})



#문제
#단어와 단어 길이로 구성된 튜플을 갖는 리스트를 만드세요.
#[('John',4), ('works',5), ('at',2), ('Intel', 5)]
a = [i for i in text.split()]
b = [len(i) for i in text.split()]

print([(a[i],b[i]) for i in range(len(a))])
#zip을 사용한 경우.
print([i for i in zip(a,b)])
print(list(zip(a,b)))

print([(i,len(i)) for i in text.split()])
print('-' * 30)

#ditionary comprehension 만들기
print({i:len(i) for i in text.split()})

#문제
#d의  key, value를 바꾼 딕셔너리를 만드세요.
d = {i:len(i) for i in text.split()}
#value를 표기하는 방법  d[k]

print([(k, d[k]) for k in d])

print({d[k]:k for k in d})
#items()사용
print({v:k for k, v in d.items()})
#
# for i in d:
#     print(i)

for i in d.items():
    print(i)


