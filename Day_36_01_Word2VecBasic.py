# Day_36_01_Word2VecBasic.py
# onehotvec 클래스의 수만큼 숫자로 단어를 볂솬 - 현실적으로 클래스의 개수가 너무 많다.
#
'''
skipgram : 

'''''
# end 위치를 구하세요.
# 전체위치에서 target범위만 제거하세요.
def extrast(token_count, target, window_size ):
    start = max(target - window_size, 0)
    end = min(target + window_size + 1, token_count)
    return [i for i in range(start, end) if i != target]


def show_dataset(tokens, window_size, is_skipgram):
    token_count = len(tokens)
    for target in range(token_count):
        surround = extrast(token_count, target, window_size)
        print(target, surround, end='')

        # 문제
        # surround가 가라키는 단어들을 출력하세요.

        if is_skipgram:
            # print(list([zip([target] * len(surround), surround)]))
            print([(tokens[t], tokens[s]) for t, s in zip([target] * len(surround), surround)])
        else:
            print([tokens[i] for i in surround], tokens[target])



tokens = ['the', 'quick', 'brown', 'fax','jumps','over', 'the', 'lazy', 'dog']

# show_dataset(tokens, 1, is_skipgram=True)
# # show_dataset(tokens, 1, is_skimgram= False )

show_dataset(tokens, 2, is_skipgram=True)
print()
show_dataset(tokens, 2, is_skipgram=False)