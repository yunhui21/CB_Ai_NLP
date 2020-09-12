# Day_04_04_csv.py
import csv

# csv : Comma separated values


def read_csv_1():
    f = open('data/sample.csv', 'r', encoding='utf-8')

    for line in f:
        print(line.strip().split(','))

    f.close()


def read_csv_2():
    f = open('data/us-500.csv', 'r', encoding='utf-8')

    rows = []
    for line in csv.reader(f):
        # print(line)
        rows.append(line)

    f.close()
    return rows


def read_csv_3(file_path):
    f = open(file_path, 'r', encoding='utf-8')

    rows = []
    for line in csv.reader(f):
        # print(line)
        rows.append(line)

    f.close()
    return rows


def write_csv(rows):
    f = open('data/us-sample.csv', 'w',
             encoding='utf-8',
             newline='')

    # 문제
    # rows를 파일에 저장하세요
    # for row in rows:
    #     # print(row)
    #     # f.write(str(row))
    #     # f.write('\n')
    #     for col in row:
    #         f.write('"{}"'.format(col))
    #         f.write(',')
    #     f.write('\n')

    writer = csv.writer(f,
                        quoting=csv.QUOTE_ALL,
                        delimiter=':')
    # for row in rows:
    #     writer.writerow(row)

    writer.writerows(rows)

    f.close()


if __name__ == '__main__':
    # read_csv_1()
    rows = read_csv_2()
    write_csv(rows)

    print('======4-4========')

# print(__name__)



