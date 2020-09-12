# Day_03_03_file.py


def read_1():
    f = open('data/poem.txt', 'r', encoding='utf-8')    # euc-kr

    lines = f.readlines()
    print(lines)
    print(type(lines))      # <class 'list'>

    # 문제
    # 파일 데이터를 줄 단위로 출력하세요
    # for line in lines:
    #     print(line, end='')

    for line in lines:
        print(line.strip())

    f.close()


def read_2():
    f = open('data/poem.txt', 'r', encoding='utf-8')

    while True:
        line = f.readline()
        # print(len(line))

        # 거짓 : False, 0, 0.0, None, [], ''
        # if len(line) == 0:
        if not line:
            break

        print(line.strip())

    f.close()


def read_3():
    f = open('data/poem.txt', 'r', encoding='utf-8')

    # iterable : range, [], '', file
    for line in f:
        print(line.strip())

    f.close()


def read_4():
    with open('data/poem.txt', 'r', encoding='utf-8') as f:
        for line in f:
            print(line.strip())

        # f.close()


def write_1():
    f = open('data/sample.txt', 'w', encoding='utf-8')

    # 문제
    # 개행 문자를 출력해서 문자별로 줄바꿈을 해주세요
    f.write('hello')
    f.write('\n')
    f.write('python')

    f.close()


def write_2():
    f1 = open('data/poem.txt', 'r', encoding='utf-8')
    # lines = f1.readlines()
    text = f1.read()
    f1.close()

    f2 = open('data/sample.txt', 'w', encoding='utf-8')
    # f2.writelines(lines)
    f2.write(text)
    f2.close()


def write_3(source, target):
    f1 = open(source, 'r', encoding='utf-8')
    f2 = open(target, 'w', encoding='utf-8')

    # lines = f1.readlines()
    # f2.writelines(lines)

    for line in f1:
        f2.write(line)

    f1.close()
    f2.close()


# read_1()
# read_2()
# read_3()
# read_4()

# write_1()
# write_2()
write_3('data/poem.txt', 'data/sample.txt')

# 문제
# 파일 복사 함수를 만드세요 (write_2)

# print('{}'.format(12))
# print('{} {}'.format(12, 3.14))
# print('{} : {}'.format(12, 3.14))
#
# s1 = str(12)
#
# s = '\n\n\t\t   AA BB CC  \t\t\n\n'
# print('[{}]'.format(s))
# print('[{}]'.format(s.strip()))
