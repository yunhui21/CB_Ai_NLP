# Day_05_03_exception.py

# a = [1, 3, 5]
# print(a[len(a)])

try:
    a = [1, 3, 5]
    print(a[len(a)])
    print(a[0], a[1], a[2])
except IndexError as e:
    print('IndexError')
    print(e)


# 문제
# 아래 코드에서 발생하는 예외를 처리하세요
try:
    b = '123hello'
    c = int(b)
    print(c)
except ValueError as e:
    print(e)
except:
    print('unknown')


while True:
    try:
        a = input('number : ')
        a = int(a)
        break
    except ValueError:
        print('Numbers only!')

print(a ** 2)
