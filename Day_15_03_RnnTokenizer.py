# Day_15_03_RnnTokenizer.py
import tensorflow as tf

long_text = (
    "If you want to build a ship, don't drum up people to collect wood"
    " and don't assign them tasks and work, but rather teach them"
    " to long for the endless immensity of the sea."
    )

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10)
tokenizer.fit_on_texts(long_text.split())
print(tokenizer)
print(tokenizer.index_word)

print(tokenizer.index_word[9])
print(tokenizer.index_word[10])
print(tokenizer.index_word[11])

# 9, 10, 11번째의 문자열을 매개변수로 전달하세요.
print(tokenizer.texts_to_sequences(['build', 'a', 'ship']))

sequences = [[7,8,9,1,2], [3,5,1], [12,15]]
print(tokenizer.sequences_to_texts(sequences))

print(tf.keras.preprocessing.sequence.pad_sequences(sequences))
# print(tokenizer.index_word[0])
print(tokenizer.index_word[1])
print(tokenizer.oov_token)      # out of vocabuary

print(tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='pre'))
print(tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post'))

print(tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=4))

