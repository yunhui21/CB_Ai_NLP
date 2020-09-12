# Day_08_01_Movies.py
import pandas as pd
import numpy as np


def get_ratings():
    users = pd.read_csv('data/users.dat',
                        delimiter='::',
                        engine='python',
                        header=None,
                        names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'])

    movies = pd.read_csv('data/movies.dat',
                         delimiter='::',
                         engine='python',
                         header=None,
                         names=['MovieID', 'Title', 'Genres'])

    ratings = pd.read_csv('data/ratings.dat',
                          delimiter='::',
                          engine='python',
                          header=None,
                          names=['UserID', 'MovieID', 'Rating', 'Timestamp'])

    data = pd.merge(pd.merge(ratings, users), movies)
    # print(data)
    # print(data.columns)

    return data


def pivot_basic():
    df = get_ratings()

    # 문제
    # 남자와 여자 중에 누가 평점이 높을까요?
    p1 = df.pivot_table(values='Rating', index='Gender')
    print(type(p1))
    print(p1.head(), end='\n\n')

    p2 = df.pivot_table(values='Rating', columns='Gender')
    print(p2.head(), end='\n\n')

    # 문제
    # 성별/연령별 평점을 보여주세요
    p3 = df.pivot_table(values='Rating', index='Age', columns='Gender')
    print(p3.head(), end='\n\n')

    p3.index = ["Under 18", "18-24", "25-34", "35-44", "45-49", "50-55", "56+"]
    print(p3.head(), end='\n\n')

    # 문제
    # 18살 구간의 여자 평점을 출력하세요 (2가지)
    print(p3.iloc[1])
    print(p3.iloc[1][0])
    print(p3.iloc[1, 0])
    print(p3.iloc[1]['F'], end='\n\n')

    print(p3.loc["18-24"])
    print(p3.loc["18-24"][0])
    # print(p3.loc["18-24", 0])     # error
    print(p3.loc["18-24", 'F'])
    print(p3.loc["18-24"]['F'], end='\n\n')

    print(p3['F'])
    print(p3['F'][1])
    # print(p3['F', 1])             # error
    # print(p3['F', '18-24'])       # error
    print(p3['F']['18-24'], end='\n\n')

    # p4 = df.pivot_table(values='Rating', index='Age', columns=['Gender', 'Occupation'])
    p4 = df.pivot_table(values='Rating', index='Age',
                        columns=['Occupation', 'Gender'],
                        fill_value=0)
    print(p4.head(), end='\n\n')

    print(p4.stack(), end='\n\n')
    print(p4.stack().unstack(), end='\n\n')

    p5 = df.pivot_table(values='Rating', index='Age', columns='Gender',
                        aggfunc=['mean', np.sum])
    print(p5.head(), end='\n\n')

    p5_1 = df.pivot_table(values='Rating', index='Age', columns='Gender',
                        aggfunc=np.mean)
    print(p5_1.head(), end='\n\n')

    p5_2 = df.pivot_table(values='Rating', index='Age', columns='Gender',
                        aggfunc=np.sum)
    print(p5_2.head(), end='\n\n')

    print(pd.concat([p5_1, p5_2], axis=0), end='\n\n')  # 수직
    print(pd.concat([p5_1, p5_2], axis=1), end='\n\n')  # 수평


def get_index_500():
    df = get_ratings()

    # by_title = df.groupby(by='Title')
    #
    # for g in by_title:
    #     print(g)

    by_title = df.groupby(by='Title').size()
    print(by_title, end='\n\n')

    # 문제
    # '제로 이펙트'의 평점 갯수를 알려주세요
    print(by_title[-4])
    print(by_title['Zero Effect (1998)'], end='\n\n')

    bools = (by_title >= 500)
    print(bools, end='\n\n')

    # 문제
    # 평점 500개 이상의 영화만 추출하세요
    p3 = df.pivot_table(values='Rating', index='Title', columns='Gender')
    print(p3.head(), end='\n\n')

    index_500 = p3[bools]
    print(index_500, end='\n\n')

    return index_500


# pivot_basic()

# 여성들이 가장 좋아하는 영화 top5
# 0. 일정 갯수 이상의 평점이 있는 영화 추출
# 1. 영화 제목별 평점
# 2. 여성 평점별 정렬

index_500 = get_index_500()

# by_female = index_500.sort_values(by='F')
by_female = index_500.sort_values(by='F', ascending=False)
print(by_female, end='\n\n')

# 문제
# 남녀 호불호가 갈리지 않는 영화 top5를 알려주세요
# 1. 성별 평점 차이를 계산한다
# 2. 평점 차이를 컬럼에 저장한다
# 3. 평점 차이에 따라 정렬한다
index_500['Diff'] = (index_500.F - index_500.M).abs()
print(index_500, end='\n\n')

by_diff = index_500.sort_values(by='Diff')
print(by_diff)
