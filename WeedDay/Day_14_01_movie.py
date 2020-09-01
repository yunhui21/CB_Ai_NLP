# Day_14_01_movie.py
import pandas as pd
import numpy as np


def get_data():
    users = pd.read_csv('ml-1m/users.dat',
                        header=None,
                        sep='::',
                        engine='python',
                        names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'])
    movies = pd.read_csv('ml-1m/movies.dat',
                         header=None,
                         sep='::',
                         engine='python',
                         names=['MovieID', 'Title', 'Genres'])
    ratings = pd.read_csv('ml-1m/ratings.dat',
                          header=None,
                          sep='::',
                          engine='python',
                          names=['UserID', 'MovieID', 'Rating', 'Timestamp'])

    # print(users)
    # print(movies)
    # print(ratings)

    data = pd.merge(pd.merge(ratings, users), movies)
    # print(data)

    return data


def pivot_basic():
    df = get_data()

    pv1 = df.pivot_table(index='Age',
                         values='Rating')
    print(pv1.head(), end='\n\n')

    pv2 = df.pivot_table(index='Age',
                         columns='Gender',
                         values='Rating')
    print(pv2.head(), end='\n\n')

    # 문제
    # 18세의 여성 데이터만 출력하세요 (2가지)
    # print(pv2.F, end='\n\n')
    print(pv2.F[18], end='\n\n')

    # print(pv2.loc[18], end='\n\n')
    print(pv2.loc[18][0], end='\n\n')

    pv3 = df.pivot_table(index=['Age', 'Gender'],
                         values='Rating')
    print(pv3.head(), end='\n\n')

    print(pv3.unstack().head(), end='\n\n')
    print(pv3.unstack().stack().head(), end='\n\n')

    # 문제
    # 18세의 여성 데이터만 출력하세요 (2가지)
    # print(type(pv3), end='\n\n')
    # print(pv3.loc[18, 'F'], end='\n\n')
    print(pv3.loc[18, 'F'].values, end='\n\n')

    # print(pv3['Rating'], end='\n\n')
    print(pv3.Rating[18, 'F'], end='\n\n')

    pv4 = df.pivot_table(index='Age',
                         columns=['Occupation', 'Gender'],
                         values='Rating')
    print(pv4.head(), end='\n\n')

    pv5 = df.pivot_table(index='Age',
                         columns=['Occupation', 'Gender'],
                         values='Rating',
                         fill_value='0')
    pv5.index = ["Under 18", "18-24", "25-34", "35-44", "45-49", "50-55", "56+"]
    print(pv5.head(), end='\n\n')

    pv6 = df.pivot_table(index='Age',
                         columns='Gender',
                         values='Rating',
                         aggfunc=[np.mean, np.sum])  # 'sum'
    print(pv6.head(), end='\n\n')

    pv7 = df.pivot_table(index='Age',
                         columns='Gender',
                         values='Rating',
                         aggfunc=np.mean)

    pv8 = df.pivot_table(index='Age',
                         columns='Gender',
                         values='Rating',
                         aggfunc=np.sum)

    pv9 = pd.concat([pv7, pv8], axis=0)
    print(pv9, end='\n\n')


def over500():
    df = get_data()

    # 1. 영화 제목별 남녀 평점
    by_title = df.pivot_table(index='Title',
                              columns='Gender',
                              values='Rating')
    print(by_title.head(), end='\n\n')

    # 2. 최소 500번 이상 평가한 영화 목록
    by_count = df.groupby(by='Title').size()
    # for t in by_count:
    #     print(t)

    print(by_count, end='\n\n')
    print(type(by_count))

    # over_500 = (by_count.values >= 500)     # broadcast
    bool_500 = (by_count >= 500)
    print(bool_500, end='\n\n')

    over_500 = by_count[bool_500]
    print(over_500, end='\n\n')

    return by_title, over_500
    # return by_title.loc[over_500.index]



# pivot_basic()

# 문제
# by_title에서 titles와 일치하는 영화만 추출하세요
by_title, titles = over500()

title_500 = by_title.loc[titles.index]
print(title_500, end='\n\n')

# 문제
# 여성들이 선호하는 영화 top5를 찾아보세요
# top_female = title_500.sort_values('F')
# print(top_female.tail(), end='\n\n')

top_female = title_500.sort_values('F', ascending=False)
print(top_female.head(), end='\n\n')

# 문제
# 성별 호불호가 갈리지 않는 영화 top5를 찾아보세요
title_500['Diff'] = (title_500.F - title_500.M).abs()
print(title_500.head(), end='\n\n')

diff_500 = title_500.sort_values('Diff')
print(diff_500.head(), end='\n\n')


# 1. 영화 제목별 평점
# 2. 최소 500번 이상 평가한 영화 목록
# 3. 여성들이 선호하는 영화 검색
