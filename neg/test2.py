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
model_name_pos2 = './100features_100minwords_2text_pos2'  # 이 모델을 사용
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
max = 5929
"""for i in range(1, 40001):

    if i % 400 == 0:
        print(i // 400, '%')

    if os.path.exists('./../../data/movie_review/train_movie_review%d.txt' % i):
        f = open('./../../data/movie_review/train_movie_review%d.txt' % i, 'r', encoding='utf-8')
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
vocab_size = 396756
train_x = np.load('../Paper/learning/word_train_x.npy')
train_y = np.load('../Paper/learning/word_train_y.npy')
print(train_x.shape, train_y.shape)
re_train_y = np.zeros((len(train_y), 1))
re_train_y = train_y[:len(train_y), :]
# train_x = train_x.reshape(vocab_size // 2, 2, 100)
train_x = train_x.reshape(vocab_size // 2, 200)
ont_hot_train_y = tf.squeeze(tf.one_hot(re_train_y, 2), axis=1)

print(train_x.shape, ont_hot_train_y.eval().shape)

X = tf.placeholder(tf.float32, [None, 200])
X1 = tf.reshape(X, [-1, 10, 10, 2])
Y = tf.placeholder(tf.float32, [None, 2])
keep_prob = tf.placeholder(tf.float32)

W1 = tf.get_variable('W1', [200, 800], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([800]))
L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), b1))

W2 = tf.get_variable('W2', [800, 800], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([800]))
L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), b2))

W3 = tf.get_variable('W3', [800, 800], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([800]))
L3 = tf.nn.relu(tf.add(tf.matmul(L2, W3), b3))

W4 = tf.get_variable('W4', [800, 800], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([800]))
L4 = tf.nn.relu(tf.add(tf.matmul(L3, W4), b4))

W5 = tf.get_variable('W5', [800, 800], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([800]))
L5 = tf.nn.relu(tf.add(tf.matmul(L4, W5), b5))

W6 = tf.get_variable('W6', [800, 800], initializer=tf.contrib.layers.xavier_initializer())
b6 = tf.Variable(tf.random_normal([800]))
L6 = tf.nn.relu(tf.add(tf.matmul(L5, W6), b6))

W7 = tf.get_variable('W7', [800, 800], initializer=tf.contrib.layers.xavier_initializer())
b7 = tf.Variable(tf.random_normal([800]))
L7 = tf.nn.relu(tf.add(tf.matmul(L6, W7), b7))

W8 = tf.get_variable('W8', [800, 800], initializer=tf.contrib.layers.xavier_initializer())
b8 = tf.Variable(tf.random_normal([800]))
L8 = tf.nn.relu(tf.add(tf.matmul(L7, W8), b8))

W9 = tf.get_variable('W9', [800, 800], initializer=tf.contrib.layers.xavier_initializer())
b9 = tf.Variable(tf.random_normal([800]))
L9 = tf.nn.relu(tf.add(tf.matmul(L8, W9), b9))

W10 = tf.get_variable('W10', [800, 2], initializer=tf.contrib.layers.xavier_initializer())
b10 = tf.Variable(tf.random_normal([2]))
hypothesis = tf.matmul(L9, W10) + b10


# cost = tf.reduce_mean(-tf.reduce_mean(Y * tf.log(hypothesis), axis=1))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)


sess = tf.Session()
sess.run(tf.global_variables_initializer())


training_epoch = 50
batch_size = 100
global_steps = 1

for epoch in range(training_epoch):
    avg_cost = 0
    steps = int((vocab_size / 2) / batch_size)

    for step in range(steps):
        batch_xs, batch_ys = next_batch(100, train_x, ont_hot_train_y.eval())
        cost_val, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.7})

        avg_cost += cost_val / steps

    print("Epoch: ", "%04d" % (epoch + 1), 'cost: ', '{:3f}'.format(avg_cost))

    batch_xs, batch_ys = next_batch(30000, train_x, ont_hot_train_y.eval())
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy: ", sess.run(accuracy, feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 1.0}))

batch_xs, batch_ys = next_batch(160000, train_x, ont_hot_train_y.eval())
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Accuracy: ", sess.run(accuracy, feed_dict={X: batch_xs, Y: batch_ys}))

saver = tf.train.Saver()
save_path = saver.save(sess, './word_learning_result4.ckpt')