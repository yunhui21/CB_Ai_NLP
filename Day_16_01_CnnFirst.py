'''
1. Convolution 은 행렬곱셈
2. 필터(kernel)은 weight(가중치)
    depth = 원본이미지의 차원을 갖는다.
3. one_number = w
4. 라츠보 -
5. 조대협..(구글 엔지니어링)
6. filter는 가장 비슷한 이미지를 추출한다.
7. weight는 학습에 의해 더 좋은 값을 가진다.
8. strid, zero-padding, channerl
10. padding : 입력레이어는 영향이 적지만 레이어가 깊어질수록 영향이 커진다.
11. 크기는 작아지지만 depth는 커진다.
12. zero padding은 원하는 만큼 값을 넣어서 사용
13. convnet = 보여지는 이미지는 weight를 보여준것.
14. local imvariance
15.
'''

# Day_16_01_CnnFirst.py
import tensorflow as tf

(x_train, y_train ) , (x_test, y_test) = tf.keras.datasets.mnist.load_data()

w1 = tf.Variable(tf.random.normal([3, 3, 1, 32])) # weight * height * depth(3, 3, 1)
b1 = tf.Variable(tf.zeros([32])) # class, feature 개수

w2 = tf.Variable(tf.random.normal([3, 3, 32, 64])) # 앞서 입력된 채널값이어야 한다.
b2 = tf.Variable(tf.zeros([64])) # 크기가 작아서 가능하다.

# -------------------------------------------------------- #

ph_x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
ph_y = tf.placeholder(tf.int32)

c1 = tf.nn.conv2d(ph_x, filter = w1, strides = [1,1,1,1], padding = 'SAME')
r1 = tf.nn.relu(c1 + b1)
p1 = tf.nn.max_pool2d(r1, ksize=[1, 2, 2, 1], strides= [1, 2, 2, 1], padding= 'SAME') # 14*14

c2 = tf.nn.conv2d(p1, filter = w2, strides = [1,1,1,1], padding = 'SAME')
r2 = tf.nn.relu(c2 + b2)
p2 = tf.nn.max_pool2d(r2, ksize=[1, 2, 2, 1], strides= [1, 2, 2, 1], padding= 'SAME')

print(c1.shape)
print(p1.shape)

print(c2.shape)
print(p2.shape)
