# Day_09_03_LinearRegressionCars.py
import tensorflow as tf
import matplotlib.pyplot as plt

# 문제
# 속도가 30과 50일 때의 제동거리를 알려주세요


def get_data():
    f = open('data/cars.csv', 'r', encoding='utf-8')
    f.readline()

    speed, dist = [], []
    for line in f:
        # print(line.strip().split(','))
        items = line.strip().split(',')

        speed.append(int(items[1]))
        dist.append(int(items[2]))

    f.close()
    return speed, dist


def linear_regression_cars():
    # x = [1, 2, 3]
    # y = [1, 2, 3]
    x, y = get_data()

    w = tf.Variable(tf.random.uniform([1]))
    b = tf.Variable(tf.random.uniform([1]))

    hx = w * x + b

    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.001)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train)
        print(i, sess.run(loss))

    sess.close()


linear_regression_cars()






