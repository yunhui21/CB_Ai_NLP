import random
import string
import numpy as np
#DAy_01_01_review.py
#ctrl+shift+f10
#alt+1
#alt+4

#문제_01
#문자열 기스트에서 가장 긴 문자열의 긴 문자열의 길이를 구하세요
#(컴프리헨션) 사용 list내표
a = ['Guest', 'Detroit', 'Woodlawn', 'Cemetary', 'People']
for i in a:
    print(len(a))

print([len(i) for i in a])
print(max([len(i) for i in a]))

a = ['Guest', 'Detroit', 'Woodlawn','Demetrary','People']
for i in a :
    print(len(a))
#문제_02
#100보다 작은 난수 10개를 갖는 리스트를 만드세요
# numbers=[]
#for _ in range(10):
#    numbers.append(random.randrange(100))
#print(numbers)

numbers = [random.randint(1,100) for _ in range(10)]
print(numbers)

#문제_03
#양수 리스트의 요소를 거꾸로 뒤십으세요
#[31, 46, 72]=>[13,64,27]
#30 : 3 * 10 + 1
#13 : 1 * 10 + 3

# x = [31,46,71]
# y =reversed(x)
# print(list(y))

print([(i//10, i%10) for i in numbers])
print([i//10 + i%10*10 for i in numbers])
print('-' * 30)

#문제_04
#2차원 리스트를 1차원으로 바꾸세요.(for문 2번 사용)
alphabest = list(string.ascii_lowercase)
print(alphabest)

letters = [['a', 'b', 'c', 'd', 'e', 'f'],
           ['g', 'h', 'i', 'j', 'k', 'l'],
           ['m', 'n', 'o', 'p', 'q', 'r'],
           ['s', 't', 'u', 'v', 'w', 'x']]
#정답2
for s in letters:
    # print(s)
    for c in s:
        print(c, end='')
print()

#정답2
print([c for s in letters for c in s])

print(letters)

#파이썬을 파이썬답게 자료
print([e for array in letters for e in array])

#문제
#2차원 난수 리스트를 만들고
#홀수 합계가 가장 큰행의 위치(인텍스)를 알려주세요.
t= [[random.randrange(6)] for j in range(4)]
print(*t, sep='\n')
print([[j for j in i if j % 2] for i in t])
print([sum[[j for j in i if j % 2] for i in t])
print(max([sum[[j for j in i if j % 2] for i in t]))
print(np.argmax([sum[[j for j in i if j % 2] for i in t]))




my_numbers= [[random.randrange(100) for j in range(5)] for i in range(20)]
#[]안에 [] 차원을 줄수가 있다.
print(*my_numbers, sep='\n')