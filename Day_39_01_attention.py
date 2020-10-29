# Day_39_01_attention.py
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np



def build_model(input_dims):
    inputs = tf.keras.layers.Input([input_dims])
    att_prob = tf.keras.layers.Dense(input_dims, activation='softmax')(inputs)

    att_mul = tf.keras.layers.multiply([inputs, att_prob])

    dense_mul = tf.keras.layers.Dense(64)(att_mul)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(dense_mul)

    return tf.keras.Model(inputs, outputs)

def get_data(n, input_dims, att_column):
    x = np.random.standard_normal(size=[n, input_dims])
    y = np.random.choice([0, 1], size=(n, 1)) # 0과 1중에서 고르겠다.
    print(x.shape, y.shape) # (10000, 32) (10000, 1)
    print(y[:10])           # [[0], [0], [0], [1], [0], [0], [0], [1], [0], [0]]


    x[:, att_column] = y[:, 0] # y.reshape[-1] # keyword

    return x, y

input_dims = 32
model = build_model(input_dims)
model.summary()

x_train, y_train = get_data(n = 10000, input_dims=input_dims, att_column=7)
x_test, y_test = get_data(n = 10000, input_dims=input_dims, att_column=7)

model.compile(optimizer='adam',
              loss=tf.keras.losses.binary_crossentropy,
              metrics=['acc'])
model.fit(x_train, y_train, epochs= 20, batch_size=64, validation_split=0.5, verbose=2)

print('acc:', model.evaluate(x_test, y_test))

preds = model.predict(x_test)
print(preds.shape)  # (10000, 1)

#------------------------------------------------------------------------------------#
layer_outputs = [layer.output for layer in model.layers]
act_model = tf.keras. Model(model.input, layer_outputs)

preds_outputs = act_model.predict(x_test)
print(len(preds_outputs), len(layer_outputs))  # 5 5 model안의 weghit 값을 꺼내기 위한.
print(preds_outputs[1].shape) # (10000, 32)

result = np.argmax(preds_outputs[1], axis=1)
print(result) # [7 9 7 ... 7 7 7]
print(list(result).count(7)) # 9680

activations = np.mean(preds_outputs[1], axis=0)
print(activations.shape)    # (32,)

plt.bar(range(len(activations)), activations)
plt.show()
