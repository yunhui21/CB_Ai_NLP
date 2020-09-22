# 071-080_tuple.py

# 071
my_variable = ()
print('071:', my_variable, type(my_variable))

# 072
movie_rank = ('닥터스트레인지', '스플릿', '럭키')
print('072:', movie_rank)

# 073
nums = (1,)
print('073:', type(nums))

# 074
t = (1,2,3)
# t[0] = 'a'
# print('074:',t[0]) # 에러 : 튜플은 값의 변환이 안된다.

# 075
t = 1,2,3,4
print('075:', t, type(t)),

# 076
t = ('a', 'b', 'c')
t = ('A', 'B', 'C')
print('076:', t)

# 077
interest = ('삼성전자', 'LG전자', 'SK Hynix')
data = list(interest)
print('077:', data)

# 078
interest = ['삼성전자', 'LG전자', 'SK Hynix']
data = tuple(interest)
print('078:', data)

# 079
temp = ('apple', 'banana', 'cake')
a, b, c = temp
print('079:', a, b, c)

# 080
nums = tuple(range(2, 99, 2))
print('080:', nums)