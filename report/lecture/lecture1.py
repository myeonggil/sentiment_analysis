import tensorflow as tf

hello = tf.constant("Hello, TensorFlow!")
sess = tf.Session()
print(sess.run(hello))

a = tf.constant(3.0, tf.float32)
b = tf.constant(4.0)
c = tf.add(a, b)

print(sess.run(c))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
add_node = a + b

print(sess.run(add_node, feed_dict={a: 3, b: 4.5}))
print(sess.run(add_node, feed_dict={a: [1, 2], b: [2, 4]}))