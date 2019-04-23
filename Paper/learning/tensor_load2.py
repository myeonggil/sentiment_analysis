import tensorflow as tf
import numpy as np
from konlpy.tag import *
import gensim.models as g
import matplotlib.pyplot as plt
import os
import random

np.seterr(divide='ignore', invalid='ignore')
sess = tf.InteractiveSession()


X = tf.placeholder(tf.float32, [None, 200])
Y = tf.placeholder(tf.float32, [None, 2])
keep_prob = tf.placeholder(tf.float32)

W1 = tf.get_variable('W1', [200, 1600], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([1600]))
L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), b1))
L1 = tf.nn.dropout(L1, keep_prob)

W2 = tf.get_variable('W2', [1600, 1600], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([1600]))
L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), b2))
L2 = tf.nn.dropout(L2, keep_prob)

W3 = tf.get_variable('W3', [1600, 1600], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([1600]))
L3 = tf.nn.relu(tf.add(tf.matmul(L2, W3), b3))
L3 = tf.nn.dropout(L3, keep_prob)

W4 = tf.get_variable('W4', [1600, 1600], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([1600]))
L4 = tf.nn.relu(tf.add(tf.matmul(L3, W4), b4))
L4 = tf.nn.dropout(L4, keep_prob)

W5 = tf.get_variable('W5', [1600, 1600], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([1600]))
L5 = tf.nn.relu(tf.add(tf.matmul(L4, W5), b5))
L5 = tf.nn.dropout(L5, keep_prob)

W6 = tf.get_variable('W6', [1600, 1600], initializer=tf.contrib.layers.xavier_initializer())
b6 = tf.Variable(tf.random_normal([1600]))
L6 = tf.nn.relu(tf.add(tf.matmul(L5, W6), b6))
L6 = tf.nn.dropout(L6, keep_prob)

W7 = tf.get_variable('W7', [1600, 1600], initializer=tf.contrib.layers.xavier_initializer())
b7 = tf.Variable(tf.random_normal([1600]))
L7 = tf.nn.relu(tf.add(tf.matmul(L6, W7), b7))
L7 = tf.nn.dropout(L7, keep_prob)

W8 = tf.get_variable('W8', [1600, 1600], initializer=tf.contrib.layers.xavier_initializer())
b8 = tf.Variable(tf.random_normal([1600]))
L8 = tf.nn.relu(tf.add(tf.matmul(L7, W8), b8))
L8 = tf.nn.dropout(L8, keep_prob)

W9 = tf.get_variable('W9', [1600, 1600], initializer=tf.contrib.layers.xavier_initializer())
b9 = tf.Variable(tf.random_normal([1600]))
L9 = tf.nn.relu(tf.add(tf.matmul(L8, W9), b9))
L9 = tf.nn.dropout(L9, keep_prob)

W10 = tf.get_variable('W10', [1600, 2], initializer=tf.contrib.layers.xavier_initializer())
b10 = tf.Variable(tf.random_normal([2]))
hypothesis = tf.matmul(L9, W10) + b10
# hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, './word_learning_result3.ckpt')

nlp = Twitter()
model_name_pos2 = './../../neg/100features_100minwords_2text_pos2'  # 이 모델을 사용
model_pos2 = g.Doc2Vec.load(model_name_pos2)
vocab_pos2 = list(model_pos2.wv.vocab)
vocab = model_pos2[vocab_pos2]

pre_result_positive = 0
pre_result_negative = 0
next_result_positive = 0
next_result_negative = 0
index = 1

train_x = []

f = open('test_sentences.txt', 'r', encoding='utf-8')
data = str(f.read())
tagger = nlp.pos(data)
word = []

for k in range(0, len(tagger) - 1):
    if tagger[k][0] in vocab_pos2 and tagger[k + 1][0] in vocab_pos2:
        train_x.append(list(vocab[vocab_pos2.index(tagger[k][0])]))
        train_x.append(list(vocab[vocab_pos2.index(tagger[k + 1][0])]))
        # word.append(tagger[j][0])
        # word.append(tagger[j + 1][0])

embedding_size = 100


re_train_x = np.array(train_x)
re_train_x = re_train_x.reshape(re_train_x.size // 200, 200)
result = sess.run(tf.argmax(hypothesis, 1), feed_dict={X: re_train_x, keep_prob: 1.0})
pos = 0
neg = 0
pos_ag = []
neg_ag = []
for j in range(0, len(result)):
    if result[j] == 0:
        neg += 1
        neg_ag.append(result[j])
    else:
        pos += 1
        pos_ag.append(result[j])

if pos > neg:
    print('긍정 : ', pos_ag)
else:
    print('부정 : ', neg_ag)

f.close()