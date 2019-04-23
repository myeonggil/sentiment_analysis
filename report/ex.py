import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras._impl.keras.datasets.cifar10 import load_data

def next_batch(num, data, labels):
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

(x_train, y_train), (x_test, y_test) = load_data()

y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 10), axis=1)
y_test_one_hot = tf.squeeze(tf.one_hot(y_test, 10), axis=1)

print("입력 training shape: ", x_train.shape) # 50000개의 그림
print("출력 training shape: ", y_train.shape)
print("입력 test shape: ", x_test.shape)  # 10000개의 그림
print("출력 test shape: ", y_test.shape)

X = tf.placeholder(tf.float32, [None, 32, 32, 3])
Y = tf.placeholder(tf.float32, [None, 10])

keep_prob = 0.7

W1 = tf.Variable(tf.random_normal([3, 3, 3, 32], stddev=0.01))
L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
L3 = tf.reshape(L3, [-1, 4 * 4 * 128])

W4 = tf.get_variable('W4', [4 * 4 * 128, 10],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L3, W4) + b4

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

train_epoch = 15
batch_size = 100

sess = tf.InteractiveSession()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(train_epoch):
    avg_cost = 0
    total_batch = int(50000 / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = next_batch(128, x_train, y_train_one_hot.eval())
        feed_dict = {X: batch_xs, Y: batch_ys}
        cost_val, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        if i % 100 == 0:
            print("Cost: ", cost_val)

    test_xs, test_ys = next_batch(10000, x_test, y_test_one_hot.eval())
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy: ", sess.run(accuracy, feed_dict={X: test_xs, Y: test_ys}))
    print(test_ys[0])
    plt.imshow(test_xs[0])
    plt.show()