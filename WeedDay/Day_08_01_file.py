# Day_08_01_file.py
import re


def read_1():
    f = open('./data/poem.txt', 'r', encoding='utf-8')

    lines = f.readlines()
    print(lines)

    f.close()


def read_2():
    f = open('./data/poem.txt', 'r', encoding='utf-8')

    while True:
        line = f.readline()
        print(line.strip())
        # print(line, end='')
        # print(len(line))

        if not line:
        # if len(line) == 0:
            break

    f.close()


def read_3():
    f = open('./data/poem.txt', 'r', encoding='utf-8')

    lines = []
    for line in f:
        # print(line.strip())
        lines.append(line.strip())

    f.close()
    return lines


def read_4():
    with open('./data/poem.txt', 'r', encoding='utf-8') as f:
        for line in f:
            print(line.strip())


def write():
    f = open('./data/sample.txt', 'w', encoding='utf-8')

    f.write('hello\n')
    f.write('python')

    f.close()


# read_1()
# read_2()
# read_3()
# read_4()
write()

# 문제
# 파일에 들어있는 단어 갯수는 몇 개입니까?
# lines = read_3()
# print(len(lines))
#
# count = 0
# for line in lines:
#     print(line)
#     words = re.findall(r'[가-힣]+', line)
#     print(words)
#     count += len(words)
#
# print('words :', count)

