# Chapter04_for_exem.py
# 문제 4-1
for x in range(5):
    print('*', end="")

print('\n', '4-2')

# 문제 4-2
for j in range(4):
    for i in range(5):
        print('*', end='')
    print('')

print('\n', '4-3')

# 문5제 4-3
for j in range(5):
    for i in range(j+1):
        print('*', end= '')
    print('')

print('\n', '4-4')

# 문5제 4-3
for j in range(5):
    for i in range(5-j):
        print('*', end= '')
    print('')
