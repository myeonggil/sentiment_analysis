import tensorflow as tf

x_data = [[11, 1], [1, 6]]
y_data = [[0]]

X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [None, 1])

W1 = tf.Variable(tf.random_uniform([2, 5], -1.0, 1.0), name='weight1')

W2 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0), name='weight2')
W3 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0), name='weight3')
W4 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0), name='weight4')
W5 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0), name='weight5')
W6 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0), name='weight6')
W7 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0), name='weight7')
W8 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0), name='weight8')
W9 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0), name='weight9')
W10 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0), name='weight10')

W11 = tf.Variable(tf.random_uniform([5, 1], -1.0, 1.0), name='weight11')

b1 = tf.Variable(tf.zeros([5]), name='Bias1')
b2 = tf.Variable(tf.zeros([5]), name='Bias2')
b3 = tf.Variable(tf.zeros([5]), name='Bias3')
b4 = tf.Variable(tf.zeros([5]), name='Bias4')
b5 = tf.Variable(tf.zeros([5]), name='Bias5')
b6 = tf.Variable(tf.zeros([5]), name='Bias6')
b7 = tf.Variable(tf.zeros([5]), name='Bias7')
b8 = tf.Variable(tf.zeros([5]), name='Bias8')
b9 = tf.Variable(tf.zeros([5]), name='Bias9')
b10 = tf.Variable(tf.zeros([5]), name='Bias10')

b11 = tf.Variable(tf.zeros([1]), name='Bias11')

L1 = tf.nn.sigmoid(tf.matmul(X, W1) + b1)
L2 = tf.nn.sigmoid(tf.matmul(L1, W2) + b2)
L3 = tf.nn.sigmoid(tf.matmul(L2, W3) + b3)
L4 = tf.nn.sigmoid(tf.matmul(L3, W4) + b4)
L5 = tf.nn.sigmoid(tf.matmul(L4, W5) + b5)
L6 = tf.nn.sigmoid(tf.matmul(L5, W6) + b6)
L7 = tf.nn.sigmoid(tf.matmul(L6, W7) + b7)
L8 = tf.nn.sigmoid(tf.matmul(L7, W8) + b8)
L9 = tf.nn.sigmoid(tf.matmul(L8, W9) + b9)
L10 = tf.nn.sigmoid(tf.matmul(L9, W10) + b10)

hypothesis = tf.nn.sigmoid(tf.matmul(L10, W11) + b11)

# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(20001):
        cost_val, _ = sess.run([cost, train], feed_dict={X:x_data, Y: y_data})
        if step % 200 == 0:
            print("Cost: ", cost_val)

    h, c, a = sess.run([hypothesis, cost, accuracy], feed_dict={X: x_data, Y: y_data})
    print("Hypothesis: ", h, "\nCost: \n", c, "\nAccuracy:\n", a)
