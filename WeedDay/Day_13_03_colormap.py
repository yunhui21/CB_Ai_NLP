# Day_13_03_colormap.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import pandas as pd


def colormap_1():
    x = np.random.rand(100)
    y = np.random.rand(100)
    t = np.arange(100)

    # plt.plot(x, y, 'ro')
    # plt.scatter(x, y)
    plt.scatter(x, y, c=t)
    plt.show()


# 문제
# 반대쪽 대각선 그려보세요
def colormap_2():
    x = np.arange(100)

    print(cm.viridis(0))
    print(cm.viridis(255))

    # plt.scatter(x, x)
    # plt.scatter(x, x, c=x)
    # plt.scatter(x, x, c=x, cmap='jet')
    # plt.scatter(x, x[::-1], c=x, cmap='jet')
    # plt.scatter(x[::-1], x, c=x, cmap='jet')
    # plt.scatter(x[::-1], x[::-1], c=x, cmap='jet')    # 실패

    # plt.scatter(x, x, c=-x, cmap='jet')
    # plt.scatter(x, x, c=x[::-1], cmap='jet')
    plt.scatter(x, x, c=x, cmap='jet_r')
    plt.show()


def colormap_3():
    print(plt.colormaps())
    print(len(plt.colormaps()))     # 166

    # ['Accent', 'Accent_r', 'Blues', 'Blues_r',
    # 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r',
    # 'BuPu', 'BuPu_r', 'CMRmap',
    # 'CMRmap_r', 'Dark2', 'Dark2_r',
    # ...
    # 'twilight_shifted', 'twilight_shifted_r',
    # 'viridis', 'viridis_r', 'winter', 'winter_r']

    x = np.arange(100)

    plt.figure(1)
    plt.scatter(x, x, c=x, cmap='winter')
    # plt.show()

    plt.figure(2)
    plt.scatter(x, x, c=x, cmap='twilight_shifted')
    plt.colorbar()
    plt.show()


def colormap_4():
    # plt.imshow(np.random.rand(10, 10))
    plt.imshow(np.random.rand(10, 10), cmap='Accent')
    plt.tight_layout()
    plt.show()


def colormap_5():
    jet = cm.get_cmap('jet')

    print(jet(-5))
    print(jet(0))
    print(jet(127))
    print(jet(128))
    print(jet(255))
    print(jet(256))
    print()

    print(jet(0.1))
    print(jet(0.5))
    print(jet(0.7))
    print()

    print(jet(128/255))
    print(jet(255/255))
    print()

    print(jet([0, 255]))
    print(jet(range(0, 256, 32)))
    print(jet(np.linspace(0.2, 0.7, 3)))

    # print(np.arange(0, 1, 0.1))
    # print(np.linspace(0, 1, 11))


def colormap_6():
    flights = sns.load_dataset('flights')
    print(type(flights))
    print(flights, end='\n\n')

    df = flights.pivot('month', 'year', 'passengers')
    print(df, end='\n\n')

    # plt.pcolor(df)
    #
    # # plt.xticks(range(12), df.index)
    # plt.yticks(0.5 + np.arange(0, 12, 2), df.index[::2])
    # plt.xticks(0.5 + np.arange(12), df.columns)
    # plt.title('flights heatmap')
    #
    # plt.colorbar()
    # plt.show()

    # sns.heatmap(df)
    # sns.heatmap(df, annot=True, fmt='d')
    sns.heatmap(df, annot=True, fmt='d', cmap='jet')

    plt.show()


# colormap_1()
# colormap_2()
# colormap_3()
# colormap_4()
# colormap_5()
colormap_6()
