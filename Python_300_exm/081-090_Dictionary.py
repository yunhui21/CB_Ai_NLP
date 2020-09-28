# 081-090_Dictionary.py

# 081
scores = [8.8, 8.9, 8.7, 9.2, 9.3, 9.7, 9.9, 9.5, 7.8, 9.4]
*valid_score, _, _= scores
print('081:', valid_score)

# 082
a, b, *valid_score = scores
print('082:', valid_score)

# 083
a, *valid_score, b = scores
print('083:', valid_score)

# 084
temp = {}
print('084:', temp, type(temp))

# 085
icecream = {'메로나':1000, '폴라포': 1200, '빵빠레': 1800}
print('085:', icecream, type(icecream))

# 086
icecream['죠스바'] = 1200
icecream['월드콘'] = 1500
print('086:', icecream)

# 087
print('087:', '메로나 가격:', icecream['메로나'])

# 088
icecream['메로나'] = 1300
print('088:', icecream)

# 089
del icecream['메로나']
print('089:', icecream)

# 090
# 키값이 없는 것을 인덱싱하면 에러가 난다.