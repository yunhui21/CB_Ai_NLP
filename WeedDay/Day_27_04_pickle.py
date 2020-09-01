# Day_27_04_pickle.py
import pickle


def save():
    d = {'age': 23, 'name': 'kim'}

    f = open('data/dict.pkl', 'wb')     # binary mode
    pickle.dump(d, f)
    f.close()


def load():
    f = open('data/dict.pkl', 'rb')
    d = pickle.load(f)
    print(d)
    f.close()


save()
load()
