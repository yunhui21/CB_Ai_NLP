# 031-040_str연산.py

# 031
a = '3'
b = '4'
print('031:', a + b)

# 032
print('032:', 'hi'*3)

# 033
print('033:', '-'*80)

# 034
t1 = 'python'
t2 = 'java'
print('034:', (t1 +' '+t2+ ' ')*4)

# 035
name1 = '김민수'
age1 = 10
name2 = '이철희'
age2 = 12

print('035:', '이름:', name1, '나이:', age1)
print('035:', '이름:', name2, '나이:', age2)

# 036
print('036:', '이름: {} 나이: {}'.format(name1, age1))
print('036:', '이름: {} 나이: {}'.format(name2, age2))

# 037
print('037:', f'이름: {name1} 나이: {age1}')
print('037:', f'이름: {name2} 나이: {age2}')

# 038
상장주식주 = "5,969,782,550"
int = int(상장주식주.replace(',',''))
print('038:', int, type(int))

# 039
분기 = '2020/03(E) (IFRS연결)'
print('039:', 분기[:7])

# 040
data = '     삼성전자      '
print('040:', data.strip())