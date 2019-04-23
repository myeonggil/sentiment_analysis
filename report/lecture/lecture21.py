import tensorflow as tf
import numpy as np

sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")

char_set = list(set(sentence))
char_dic = {w: i for i, w in enumerate(sentence)}
print(char_set)
print(char_dic)

data_dim = len(char_set)
rnn_hidden_size = len(char_set)
num_classes = len(char_set)

dataX = []
dataY = []
sequence_length = 10
for i in range(0, len(sentence) - sequence_length):
    x_str = sentence[i:sequence_length + i]
    y_str = sentence[i + 1:i + sequence_length + 1]
    print(i, x_str, '->', y_str)

    x = [char_dic[c] for c in x_str]
    y = [char_dic[c] for c in y_str]

    dataX.append(x)
    dataY.append(y)

print(dataY)
print(dataX)
batch_size = len(dataX)

X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, sequence_length])

X_one_hot = tf.one_hot(X, num_classes)
print(X_one_hot)

cell = tf.contrib.rnn.BasicLSTMCell(rnn_hidden_size, state_is_tuple=True)
cell = tf.contrib.rnn.MultiRNNCell([cell] * 2, state_is_tuple=True)
outputs, _states = tf.nn.dynamic_rnn(cell, X_one_hot, dtype=tf.float32)

print(outputs)

X_for_softmax = tf.reshape(outputs, [-1, rnn_hidden_size])

softmax_w = tf.get_variable('softmax_w', [rnn_hidden_size, num_classes])

softmax_b = tf.get_variable('softmax_b', [num_classes])
outputs = tf.matmul(X_for_softmax, softmax_w) + softmax_b

outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])
print(outputs)
weights = tf.ones([batch_size, sequence_length])
print(weights, Y)

sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
mean_loss = tf.reduce_mean(sequence_loss)
train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(mean_loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(500):
        _, l, results = sess.run([train_op, mean_loss, outputs], feed_dict={X: dataX, Y: dataY})

        for j, result in enumerate(results):
            index = np.argmax(result, axis=1)
            print(i, j, ''.join([char_set[t] for t in index]), l)