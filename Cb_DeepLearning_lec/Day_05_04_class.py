# Day_05_04_class.py


# 클래스 : 함수 + 변수
class Info:
    def __init__(self):
        print('생성자')
        self.age = 23

    def show(self):
        print('show', self.age)


i1 = Info()
i2 = Info()
print(i1)

# Info.show(123)
Info.show(i1)
Info.show(i2)
i1.show()

print(i1.age)





