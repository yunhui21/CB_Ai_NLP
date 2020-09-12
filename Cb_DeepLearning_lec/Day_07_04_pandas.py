# Day_07_04_pandas.py
import numpy as np
import pandas as pd


def series_basic():
    s1 = pd.Series([2, 1, 5, 7])
    print(s1)

    print(s1.index)
    print(s1.values)
    print(type(s1.values))
    print()

    s1.index = ['a', 'b', 'c', 'd']
    print(s1)

    print(s1[0])
    print(s1[-1])

    print(s1['a'])
    print(s1['d'])

    # 문제
    # 슬라이싱 문법을 적용해서 가운데 2개만 출력하세요
    print(s1[1:-1])
    print(s1['b':'c'])

    print(s1.index)
    print(s1.index.values)


def dataframe_basic():
    states = {
        'city': ['pusan', 'pusan', 'pusan', 'cheju', 'cheju', 'cheju'],
        'year': [2017, 2018, 2019, 2017, 2018, 2019],
        'rain': [100, 90, 85, 95, 105, 100]
    }
    df = pd.DataFrame(states, index=list('abcdef'))
    print(df, end='\n\n')

    print(df.head(), end='\n\n')
    print(df.tail(), end='\n\n')

    print(df.head(2), end='\n\n')
    print(df.tail(2), end='\n\n')

    print(df.index, end='\n\n')
    print(df.columns, end='\n\n')
    print(df.values, end='\n\n')
    print(df.values.dtype, end='\n\n')

    print(df['rain'], end='\n\n')
    print(df.rain, end='\n\n')

    print(df.iloc[0], end='\n\n')
    print(df.loc['a'], end='\n\n')


# series_basic()
dataframe_basic()
