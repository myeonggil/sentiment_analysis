import tensorflow as tf
import numpy as np
from konlpy.tag import *
import gensim.models as g
import matplotlib.pyplot as plt
import os
import random


np.seterr(divide='ignore', invalid='ignore')
sess = tf.InteractiveSession()

"""
텐서플로우 저장
saver = tf.train.Saver()
save_path = saver.save(sess, "./파일명.ckpt")
불러오기
saver = tf.train.Saver()
saver.restore(sess, save_path)
텐서보드 사용법
scope을 통해 수행과정을 레이어마다 생성
모든 summary 병합
writer로 경로와 파일명 지정
writer에 그래프 추가
run에 summary 추가
writer에 summary와 globar step 추가
cmd창에 tensorboard --logdir=파일명 입력 
"""

nlp = Twitter()
model_name_pos2 = './../../neg/100features_100minwords_2text_neg2'  # 이 모델을 사용
model_pos2 = g.Doc2Vec.load(model_name_pos2)
vocab_pos2 = list(model_pos2.wv.vocab)
X = model_pos2[vocab_pos2]

# 리뷰로부터 단어의 embedding size를 가져와 리스트에 저장하는 부분
i = 100
train_x = []
train_y = []
count = 0
pos = 0
neg = 0
# max = 1500
max = 9470

"""for i in range(1, 60001):

    if i % 600 == 0:
        print(i // 600, '%')

    if os.path.exists('./../../data/train_data/train_movie_review%d.txt' % i):
        f = open('./../../data/train_data/train_movie_review%d.txt' % i, 'r', encoding='utf-8')
        data = str(f.readline())
        line = data.split('*'*10)
        count = 0

        if line[0] == '1':
            pos += 1
            if pos >= max:
                continue

        if line[0] != '5':
            tagger = nlp.pos(line[1])
            for j in range(0, len(tagger) - 1):
                if tagger[j][0] in vocab_pos2 and tagger[j + 1][0] in vocab_pos2:
                    count += 1

            if count != 0:
                for j in range(0, len(tagger) - 1):
                    if tagger[j][0] in vocab_pos2 and tagger[j + 1][0] in vocab_pos2:
                        train_x.append(list(X[vocab_pos2.index(tagger[j][0])]))
                        train_x.append(list(X[vocab_pos2.index(tagger[j + 1][0])]))

                for value in range(0, count):
                    dif = int(line[0])
                    train_y.append([dif])

        f.close()

train_x = np.array(train_x)
train_y = np.array(train_y)

print(train_x.shape, train_y.shape)
print(train_y[0], train_y[18000])
np.save('word_train_x', train_x)
np.save('word_train_y', train_y)"""

def next_batch(num, data, labels):

    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

embedding_size = 100
# vocab_size = 243132
# train_size = 396756
# train_size = 472496
test_size = 103876
train_size = 651562
train_x = np.load('word_train_x.npy')
train_y = np.load('word_train_y.npy')
test_x = np.load('word_test_x.npy')
test_y = np.load('word_test_y.npy')
print(train_x.shape, train_y.shape)
print(test_x.shape, test_y.shape)
re_train_y = np.zeros((len(train_y), 1))
re_train_y = train_y[:len(train_y), :]
re_test_y = np.zeros((len(train_y), 1))
re_test_y = test_y[:len(test_y), :]
# train_x = train_x.reshape(vocab_size // 2, 2, 100)
train_x = train_x.reshape(train_size // 2, 200)
test_x = test_x.reshape(test_size // 2, 200)
ont_hot_train_y = tf.squeeze(tf.one_hot(re_train_y, 2), axis=1)
ont_hot_test_y = tf.squeeze(tf.one_hot(re_test_y, 2), axis=1)

print(train_x.shape, ont_hot_train_y.eval().shape)
print(test_x.shape, ont_hot_test_y.eval().shape)

X = tf.placeholder(tf.float32, [None, 200])
X1 = tf.reshape(X, [-1, 10, 10, 2])
Y = tf.placeholder(tf.float32, [None, 2])
keep_prob = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_normal([3, 3, 2, 32], stddev=0.01))
L1 = tf.nn.conv2d(X1, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
L3 = tf.reshape(L3, [-1, 128 * 4])

W4 = tf.get_variable("W4", shape=[128 * 4, 625],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([625]))
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

W5 = tf.get_variable("W5", shape=[625, 2], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([2]))
hypothesis = tf.matmul(L4, W5) + b5

"""W1 = tf.get_variable('W1', [200, 200], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([200]))
L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), b1))
L1 = tf.nn.dropout(L1, keep_prob)

W2 = tf.get_variable('W2', [200, 200], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([200]))
L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), b2))
L2 = tf.nn.dropout(L2, keep_prob)

W = tf.get_variable('W', [200, 2], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.zeros([2]))
# hypothesis = tf.matmul(L2, W3) + b3
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)"""

"""with tf.name_scope('layer1') as scope:
    W1 = tf.get_variable('W1', [200, 128], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.random_normal([128]))
    L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), b1))

    w1_hist = tf.summary.histogram('weights1', W1)
    b1_hist = tf.summary.histogram('biases1', b1)
    layer1_hist = tf.summary.histogram('layer1', L1)

with tf.name_scope('layer2') as scope:
    W2 = tf.get_variable('W2', [128, 128], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.Variable(tf.random_normal([128]))
    L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), b2))

    w2_hist = tf.summary.histogram('weights2', W2)
    b2_hist = tf.summary.histogram('biases2', b2)
    layer2_hist = tf.summary.histogram('layer2', L2)

with tf.name_scope('layer3') as scope:
    W3 = tf.get_variable('W3', [128, 128], initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.Variable(tf.random_normal([128]))
    L3 = tf.nn.relu(tf.add(tf.matmul(L2, W3), b3))

    w3_hist = tf.summary.histogram('weights3', W3)
    b3_hist = tf.summary.histogram('biases3', b3)
    layer3_hist = tf.summary.histogram('layer3', L3)

with tf.name_scope('layer4') as scope:
    W4 = tf.get_variable('W4', [128, 128], initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.Variable(tf.random_normal([128]))
    L4 = tf.nn.relu(tf.add(tf.matmul(L3, W4), b4))

    w4_hist = tf.summary.histogram('weights4', W4)
    b4_hist = tf.summary.histogram('biases4', b4)
    layer4_hist = tf.summary.histogram('layer4', L4)

with tf.name_scope('layer5') as scope:
    W5 = tf.get_variable('W5', [128, 128], initializer=tf.contrib.layers.xavier_initializer())
    b5 = tf.Variable(tf.random_normal([128]))
    L5 = tf.nn.relu(tf.add(tf.matmul(L4, W5), b5))

    w5_hist = tf.summary.histogram('weights3', W5)
    b5_hist = tf.summary.histogram('biases3', b5)
    layer5_hist = tf.summary.histogram('layer3', L5)

with tf.name_scope('layer6') as scope:
    W6 = tf.get_variable('W6', [128, 128], initializer=tf.contrib.layers.xavier_initializer())
    b6 = tf.Variable(tf.random_normal([128]))
    L6 = tf.nn.relu(tf.add(tf.matmul(L5, W6), b6))

    w6_hist = tf.summary.histogram('weights6', W4)
    b6_hist = tf.summary.histogram('biases6', b4)
    layer6_hist = tf.summary.histogram('layer6', L4)

with tf.name_scope('layer7') as scope:
    W7 = tf.get_variable('W7', [128, 128], initializer=tf.contrib.layers.xavier_initializer())
    b7 = tf.Variable(tf.random_normal([128]))
    L7 = tf.nn.relu(tf.add(tf.matmul(L6, W7), b7))

    w7_hist = tf.summary.histogram('weights7', W7)
    b7_hist = tf.summary.histogram('biases7', b7)
    layer7_hist = tf.summary.histogram('layer7', L7)

with tf.name_scope('layer8') as scope:
    W8 = tf.get_variable('W8', [128, 128], initializer=tf.contrib.layers.xavier_initializer())
    b8 = tf.Variable(tf.random_normal([128]))
    L8 = tf.nn.relu(tf.add(tf.matmul(L3, W4), b4))

    w8_hist = tf.summary.histogram('weights8', W8)
    b8_hist = tf.summary.histogram('biases8', b8)
    layer8_hist = tf.summary.histogram('layer8', L8)

with tf.name_scope('layer9') as scope:
    W9 = tf.get_variable('W9', [128, 128], initializer=tf.contrib.layers.xavier_initializer())
    b9 = tf.Variable(tf.random_normal([128]))
    L9 = tf.nn.relu(tf.add(tf.matmul(L8, W9), b9))

    w9_hist = tf.summary.histogram('weights3', W9)
    b9_hist = tf.summary.histogram('biases3', b9)
    layer9_hist = tf.summary.histogram('layer3', L9)

with tf.name_scope('layer10') as scope:
    W10 = tf.get_variable('W10', [128, 2], initializer=tf.contrib.layers.xavier_initializer())
    b10 = tf.Variable(tf.random_normal([2]))
    hypothesis = tf.matmul(L9, W10) + b10

    w10_hist = tf.summary.histogram('weights5', W10)
    b10_hist = tf.summary.histogram('biases5', b10)
    hypothesis_hist = tf.summary.histogram('hypothesis', hypothesis)"""

# cost = tf.reduce_mean(-tf.reduce_mean(Y * tf.log(hypothesis), reduction_indices=[1]))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

# summary = tf.summary.merge_all()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# writer = tf.summary.FileWriter('./result_logs3')
# writer.add_graph(sess.graph)

training_epoch = 15
batch_size = 100
global_steps = 1

for epoch in range(training_epoch):
    avg_cost = 0
    steps = int(train_size / batch_size)

    for step in range(steps):
        batch_xs, batch_ys = next_batch(batch_size, train_x, ont_hot_train_y.eval())
        cost_val, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.8})
        # writer.add_summary(s, global_step=global_steps)
        global_steps += 1
        avg_cost += cost_val / steps
        print(cost_val)

    print("Epoch: ", "%04d" % (epoch + 1), 'cost: ', '{:3f}'.format(avg_cost))

    batch_xs, batch_ys = next_batch(200, test_x, ont_hot_test_y.eval())
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy: ", sess.run(accuracy, feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 1.0}))

batch_xs, batch_ys = next_batch(1000, test_x, ont_hot_test_y.eval())
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Accuracy: ", sess.run(accuracy, feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 1.0}))

saver = tf.train.Saver()
save_path = saver.save(sess, './word_learning_result.ckpt')