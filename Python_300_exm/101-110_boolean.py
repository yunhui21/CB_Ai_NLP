# 101-110_boolean.py

# 101
print('101:', type(True))

# 102,
print('102:', 3==5)

# 103
print('103:', 3<5)

# 104
x = 4
print('104:',1 <x <5 )

# 105
print('105:', (3==3) and (4!=3))

# 106
# print('106:', 3=>4) # error 지원하지 연산자
print('106:', 3<=4)

# 107
if 4 < 3:
    print('107:',"Hello World") # 조건성립이 되지않아서 출력되지 않는다.

# 108
if 4 < 3:
    print('108:', 'Hello World')
else:
    print('108:', 'Hi, there.')

# 109
if True:
    print('109:','1')
    print('109:','2')
else:
    print('109:', '3')
print('109:', '4')

# 110
if True:
    if False:
        print('110:','1')
        print('110:','2')
    else:
        print('110:', '3')
else:
    print('110:', '4')
print('110:', '5')