# Day_10_04_class.py


class Info:
    def __init__(self):
        print('init')
        self.age = 12   # 멤버 변수

    def show(self):     # 멤버 함수
        print('show', self.age)


i1 = Info()
i2 = Info()
i2.addr = 'cheju'
print(i1)

i1.show()
# Info.show('abc')      # error
Info.show(i1)

print(i1.age)


