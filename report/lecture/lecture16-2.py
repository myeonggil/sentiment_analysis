import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

img = mnist.train.images[0].reshape(28, 28)

sess = tf.InteractiveSession()

img = img.reshape(-1, 28, 28, 1)    # -1은 n개의 이미지 알아서 계산 color = 1
W1 = tf.Variable(tf.random_normal([3, 3, 1, 5], stddev=0.01))   # filter size 3 x 3  , 5개의 filter
conv2d = tf.nn.conv2d(img, W1, strides=[1, 2, 2, 1], padding='SAME')
pool = tf.nn.max_pool(conv2d, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # ksize = 필터사이즈
print(pool)
sess.run(tf.global_variables_initializer())
pool_img = pool.eval()
pool_img = np.swapaxes(pool_img, 0, 3)
for i, one_img in enumerate(pool_img):
    plt.subplot(1, 5, i+1), plt.imshow(one_img.reshape(7, 7), cmap='gray')
    #plt.show()