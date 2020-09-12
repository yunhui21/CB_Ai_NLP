# Day_07_02_matplotlib.py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors, font_manager, rc
from wordcloud import WordCloud

# 문제
# 2016 gdp 파일을 읽어서 막대 그래프로 표시하세요
# (top 10만 표시합니다)


def show_gdp_1():
    f = open('data/2016_GDP.txt', 'r', encoding='utf-8')

    # skip header
    f.readline()

    names, dollors = [], []
    for line in f:
        items = line.strip().split(':')
        # print(items)

        money = items[2].replace(',', '')
        # print(money)
        money = int(money)

        names.append(items[1])
        dollors.append(money)

    f.close()

    names_10 = names[:10]
    dollors_10 = dollors[:10]

    idx = np.arange(10)

    ttf = 'C:\Windows\Fonts\malgun.ttf'
    font_name = font_manager.FontProperties(fname=ttf).get_name()
    rc('font', family=font_name)

    # plt.bar(idx, dollors_10)
    plt.bar(idx, dollors_10, color=colors.TABLEAU_COLORS)
    # plt.xticks(idx, names_10)
    # plt.xticks(idx, names_10, rotation=45)
    plt.xticks(idx, names_10, rotation='vertical')
    plt.subplots_adjust(bottom=0.3)

    plt.title('2016 GDP')
    plt.xlabel('나라 이름')

    plt.tight_layout()
    plt.show()


def show_gdp_2():
    f = open('data/2016_GDP.txt', 'r', encoding='utf-8')

    # skip header
    f.readline()

    rows = []
    for line in f:
        items = line.strip().split(':')

        money = items[2].replace(',', '')
        money = int(money)

        rows.append((items[1], money))


    f.close()

    rows.sort(key=lambda t: t[1], reverse=True)

    names, dollors = [], []
    for n, d in rows:
        names.append(n)
        dollors.append(d)

    # -------------------------- #

    names_10 = names[:10]
    dollors_10 = dollors[:10]

    idx = np.arange(10)

    ttf = 'C:\Windows\Fonts\malgun.ttf'
    font_name = font_manager.FontProperties(fname=ttf).get_name()
    rc('font', family=font_name)

    # plt.bar(idx, dollors_10)
    plt.bar(idx, dollors_10, color=colors.TABLEAU_COLORS)
    # plt.xticks(idx, names_10)
    # plt.xticks(idx, names_10, rotation=45)
    plt.xticks(idx, names_10, rotation='vertical')
    plt.subplots_adjust(bottom=0.3)

    plt.title('2016 GDP')
    plt.xlabel('나라 이름')

    plt.tight_layout()
    plt.show()


def show_gdp_3():
    f = open('data/2016_GDP.txt', 'r', encoding='utf-8')

    # skip header
    f.readline()

    names, dollors = [], []
    for line in f:
        items = line.strip().split(':')

        money = items[2].replace(',', '')
        money = int(money)

        names.append(items[1])
        dollors.append(money)

    f.close()

    # np.random.shuffle(dollors)
    # print(dollors)

    indices = np.argsort(dollors)
    indices = indices[::-1]
    # print(indices)

    names = np.array(names)[indices]
    dollors = np.array(dollors)[indices]

    names_10 = names[:10]
    dollors_10 = dollors[:10]

    idx = np.arange(10)

    ttf = 'C:\Windows\Fonts\malgun.ttf'
    font_name = font_manager.FontProperties(fname=ttf).get_name()
    rc('font', family=font_name)

    # plt.bar(idx, dollors_10)
    plt.bar(idx, dollors_10, color=colors.TABLEAU_COLORS)
    # plt.xticks(idx, names_10)
    # plt.xticks(idx, names_10, rotation=45)
    plt.xticks(idx, names_10, rotation='vertical')
    plt.subplots_adjust(bottom=0.3)

    plt.title('2016 GDP')
    plt.xlabel('나라 이름')

    plt.tight_layout()
    plt.show()


def show_wordcloud():
    # ont_path = 'c:\\windows\\fonts\\NanumGothic.ttf'
    wordcloud = WordCloud(
       # font_path = font_path,
       width = 800,
       height = 800
    )

    text = open('data/king.txt', 'r', encoding='utf-8').read()
    img = wordcloud.generate(text)

    fig = plt.figure(figsize=(12, 12))
    plt.imshow(img)
    plt.axis("off")
    plt.show()
    # fig.savefig('wordcloud_without_axisoff.png')


# show_gdp_1()
# show_gdp_2()
# show_gdp_3()

show_wordcloud()
