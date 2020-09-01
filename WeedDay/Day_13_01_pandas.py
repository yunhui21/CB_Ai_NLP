# Day_13_01_pandas.py
import pandas as pd

s = pd.Series([2, 1, 5, 9])
print(s)
print(type(s))

print(s.index)
print(s.values)
print(type(s.values))

print(s[0], s[3])
# print(s[-1])

s2 = pd.Series([2, 1, 5, 9],
               index=['a', 'b', 'c', 'd'])
print(s2)

print(s2[0], s2[3])
print(s2[-1])

# 문제
# 2와 9를 출력하는 다른 코드는 무엇일까요?
print(s2['a'], s2['d'])

# 문제
# 마지막 3개의 행을 출력하세요 (2가지)
print(s2[1:])
print(type(s2[1:]))

print(s2['b':])
print(s2['b':'d'])

print(s2.values)
print(s2.values[1:])
print(s2[1:].values)
print('-' * 30)

df = pd.DataFrame({
    'year': [2018, 2019, 2020,
             2018, 2019, 2020],
    'city': ['ochang', 'ochang', 'ochang',
             'sejong', 'sejong', 'sejong'],
    'rain': [130, 150, 160,
             150, 145, 155],
})
print(df)
print(type(df))

print(df.head(), end='\n\n')
print(df.tail(), end='\n\n')

print(df.head(2), end='\n\n')
print(df.tail(2), end='\n\n')

df.info()

print(df.index)
print(df.columns)
print(df.values)
print(df.values.dtype, end='\n\n')

print(df['year'])
print(df.year)
print(type(df['year']), end='\n\n')

df.index = ['a', 'b', 'c', 'd', 'e', 'f']

print(df.iloc[0])
print(df.iloc[-1])

print(df.loc['a'])
print(df.loc['f'], end='\n\n')

# 문제
# 데이터프레임에 대해 슬라이싱 문법을 확인하세요
print(df.iloc[1:4], end='\n\n')
print(df.loc['b':'d'], end='\n\n')

print(df[1:4], end='\n\n')
print(df['b':'d'], end='\n\n')

print(df['year':'rain'], end='\n\n')    # Empty DataFrame

print(df.pivot('year', 'city', 'rain'))

# print(df.ix['a'])     # error
