# 061-070_listSlicing.py

# 061
price = ['20180728', 100,130,140,150,160,170]
print('061:', price[1:])

# 062
nums = [1,2,3,4,5,6,7,8,9,10]
print('062:', nums[::2])

# 063
print('063:', nums[1::2])

# 064
nums = [1,2,3,4,5]
print('064:', nums[::-1])

# 065
interest = ['삼성전자', 'LG전자', 'Naver']
print('065:', interest[0], interest[2])

# 066
interest = ['삼성전자', 'LG전자', 'Naver', 'SK하이닉스', '미래에셋대우']
print('066:', ' '.join(interest))

# 067
print('067:', '/'.join(interest))

# 068
print('068:', '\n'.join(interest))

# 069
string = '삼성전자/LG전자/Naver'
interest = string.split('/')
print('069:', interest)

# 070
data = [2,4,3,1,5,10,9]
data.sort()
data2 = sorted(data)
print('070:', data2)
