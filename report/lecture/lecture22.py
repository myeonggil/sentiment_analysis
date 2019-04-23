import tensorflow as tf
import numpy as np

x_data = np.array([[[1, 0, 0, 0, 0]],
                    [[1, 3, 2, 2, 2]],
                    [[2, 2, 1, 1, 2]]], dtype=np.float32)

hidden_size = 2
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
outputs, _states = tf.nn.dynamic_rnn(cell, x_data, sequence_length=[5, 3, 4], dtype=tf.float32)
sess = tf.InteractiveSession()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(outputs.eval())