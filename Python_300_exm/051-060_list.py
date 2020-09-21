# 051-060_list.py

# 051
movie_rank = ['닥터스트레인지', '스플릿', '럭키']
print('051:', movie_rank)

# 052
movie_rank.append('배트맨')
print('052:', movie_rank)

# 053
movie_rank = ['닥터스트레인지', '스플릿', '럭키', '배트맨']
movie_rank.insert(1, '슈퍼맨')
print('053:', movie_rank)

# 053
movie_rank = ['닥터스트레인지', '슈퍼맨', '스플릿', '럭키', '배트맨']
del movie_rank[3]
print('053:', movie_rank)

# 054
movie_rank = ['닥터스트레인지', '슈퍼맨', '스플릿', '배트맨']
del movie_rank[2]
del movie_rank[2]
print('054:', movie_rank)

# 056
lang1 = ['C',  'C++', 'JAVA']
lang2 = ['Python', 'Go', 'C#']
lang = lang1 +lang2
print('056:', lang)

# 057
nums = [1,2,3,4,5,6,7]
nums_max = max(nums)
print('057:', nums_max)

# 058
nums = [1,2,3,4,5]
nums_sum = sum(nums)
print('058:', nums_sum)

# 059
cook = ['피자', '김밥', '만두', '양념치킨',
        '족발', '피자', '김치만두', '쫄면',
        '쏘세지', '라면', '팥빙수', '김치전']
print('059:', len(cook))

# 060
nums = [1,2,3,4,5]
average = sum(nums)/len(nums)
print('060:', average)