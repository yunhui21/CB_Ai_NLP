# Day_09_01_csv.py
import csv


# 문제
# weather.csv 파일을 필드별로 분리해서 출력하세요 (split)
def read_csv_1():
    f = open('data/weather.csv', 'r', encoding='utf-8')

    # rows = []
    # for row in f:
    #     # print(row.strip().split(','))
    #     rows.append(row.strip().split(','))

    # 문제
    # 위의 반복문을 컴프리헨션으로 바꾸세요
    rows = [row.strip().split(',') for row in f]

    f.close()
    return rows


def read_csv_2():
    f = open('data/weather.csv', 'r', encoding='utf-8')

    rows = []
    for row in csv.reader(f):
        # print(row)
        rows.append(row)

    f.close()
    return rows


def read_us_500():
    f = open('data/us-500.csv', 'r', encoding='utf-8')

    for row in csv.reader(f):
        print(row)

    f.close()


def write_csv(rows):
    f = open('data/kma.csv', 'w',
             encoding='utf-8', newline='')

    # for row in rows:
    #     f.write(','.join(row) + '\n')

    writer = csv.writer(f,
                        delimiter=':',
                        quoting=csv.QUOTE_ALL)
    # for row in rows:
    #     writer.writerow(row)
    writer.writerows(rows)

    f.close()



# 문제
# rows에 들어있는 값을 이전 결과처럼 출력하세요

# rows = read_csv_1()
# rows = read_csv_2()
#
# for row in rows:
#     # print(row)
#     # for col in row:
#     #     print(col, end=',')
#     # print('\b')
#     print(','.join(row))
#
# read_us_500()

rows = read_csv_1()
write_csv(rows)


