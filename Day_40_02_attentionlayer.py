# Day_40_02_attentionlayer.py
from tensorflow.keras.layers import Dense, Lambda, dot, Activation, concatenate
from tensorflow.keras.layers import Layer
import tensorflow as tf

class Attention(Layer): # custom calss
    def __call__(self, hidden_states): # hidden_states가 아니라 outputs가 넘어온다.

        # 시계열 데이터가 아니므로 decoder를 정의하지 않는다.
        """
        Many-to-one attention mechanism for Keras.
        @param hidden_states: 3D tensor with shape (batch_size, time_steps, input_dim).
        @return: 2D tensor with shape (batch_size, 128)
        @author: felixhao28.
        """
        hidden_size = int(hidden_states.shape[2])

        # 1) 어텐션 스코어(Attention Score)를 구한다.
        # Inside dense layer
        #              hidden_states            dot               W            =>           score_first_part
        # (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size) 행렬곱을 하면 2차원의 행렬곱셈을 batch_size만큼
        # W is the trainable weight matrix of attention Luong's multiplicative style score
        score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(hidden_states)
        #            score_first_part           dot        last_hidden_state     => attention_weights
        # (batch_size, time_steps, hidden_size) dot   (batch_size, hidden_size)  => (batch_size, time_steps)
        h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(hidden_states) # 마지막 hiddenstate만 갖고 온다. x[:, -1, :]
        score = dot([score_first_part, h_t], [2, 1], name='attention_score') # 행렬곱   [2, 1] time_steps, hidden_size->hidden_size

        # 2) 소프트맥스(softmax) 함수를 통해 어텐션 분포(Attention Distribution)를 구한다.
        attention_weights = Activation('softmax', name='attention_weight')(score)
        # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)

        # 3) 각 인코더의 어텐션 가중치와 은닉 상태를 가중합하여 어텐션 값(Attention Value)을 구한다.
        context_vector = dot([hidden_states, attention_weights], [1, 1], name='context_vector')

        # 4) 어텐션 값과 디코더의 t 시점의 은닉 상태를 연결한다.(Concatenate)
        pre_activation = concatenate([context_vector, h_t], name='attention_output')

        # 5) 출력층 연산의 입력이 되는 s~t를 계산합니다.
        attention_vector = Dense(128, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)
        return attention_vector

class BahdanauAttention(tf.keras.Model):
      def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

      def call(self, values): # 단, key와 value는 같음
        query = Lambda(lambda x: x[:, -1, :])(values)
        # query shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # score 계산을 위해 뒤에서 할 덧셈을 위해서 차원을 변경해줍니다.
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        # return context_vector, attention_weights
        return context_vector