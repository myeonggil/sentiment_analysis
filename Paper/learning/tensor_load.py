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
"""

select = input('1 or 2: ')
if select == '1':
    X = tf.placeholder(tf.float32, [None, 200])
    Y = tf.placeholder(tf.float32, [None, 2])
    keep_prob = tf.placeholder(tf.float32)

    W1 = tf.get_variable('W1', [200, 200], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.random_normal([200]))
    L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), b1))
    L1 = tf.nn.dropout(L1, keep_prob)

    W2 = tf.get_variable('W2', [200, 200], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.Variable(tf.random_normal([200]))
    L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), b2))
    L2 = tf.nn.dropout(L2, keep_prob)

    W3 = tf.get_variable('W3', [200, 200], initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.Variable(tf.random_normal([200]))
    L3 = tf.nn.relu(tf.add(tf.matmul(L2, W3), b3))
    L3 = tf.nn.dropout(L3, keep_prob)

    W4 = tf.get_variable('W4', [200, 200], initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.Variable(tf.random_normal([200]))
    L4 = tf.nn.relu(tf.add(tf.matmul(L3, W4), b4))
    L4 = tf.nn.dropout(L4, keep_prob)

    W5 = tf.get_variable('W5', [200, 200], initializer=tf.contrib.layers.xavier_initializer())
    b5 = tf.Variable(tf.random_normal([200]))
    L5 = tf.nn.relu(tf.add(tf.matmul(L4, W5), b5))
    L5 = tf.nn.dropout(L5, keep_prob)

    W6 = tf.get_variable('W6', [200, 200], initializer=tf.contrib.layers.xavier_initializer())
    b6 = tf.Variable(tf.random_normal([200]))
    L6 = tf.nn.relu(tf.add(tf.matmul(L5, W6), b6))
    L6 = tf.nn.dropout(L6, keep_prob)

    W7 = tf.get_variable('W7', [200, 200], initializer=tf.contrib.layers.xavier_initializer())
    b7 = tf.Variable(tf.random_normal([200]))
    L7 = tf.nn.relu(tf.add(tf.matmul(L6, W7), b7))
    L7 = tf.nn.dropout(L7, keep_prob)

    W8 = tf.get_variable('W8', [200, 200], initializer=tf.contrib.layers.xavier_initializer())
    b8 = tf.Variable(tf.random_normal([200]))
    L8 = tf.nn.relu(tf.add(tf.matmul(L7, W8), b8))
    L8 = tf.nn.dropout(L8, keep_prob)

    W9 = tf.get_variable('W9', [200, 200], initializer=tf.contrib.layers.xavier_initializer())
    b9 = tf.Variable(tf.random_normal([200]))
    L9 = tf.nn.relu(tf.add(tf.matmul(L8, W9), b9))
    L9 = tf.nn.dropout(L9, keep_prob)

    W10 = tf.get_variable('W10', [200, 200], initializer=tf.contrib.layers.xavier_initializer())
    b10 = tf.Variable(tf.random_normal([200]))
    L10 = tf.nn.relu(tf.add(tf.matmul(L9, W10), b10))
    L10 = tf.nn.dropout(L10, keep_prob)

    W11 = tf.get_variable('W11', [200, 2], initializer=tf.contrib.layers.xavier_initializer())
    b11 = tf.Variable(tf.random_normal([2]))
    hypothesis = tf.matmul(L10, W11) + b11
    # hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, './word_learning_result.ckpt')

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

    for i in range(1, 10001):
        f = open('movie/1/movie_review%d.txt' % i, 'r', encoding='utf-8')
        train_x = []

        if i // 100 == index:
            print('progress - [' + str('#' * index) + str('-' * (100 - index) + '] ') + str(index) + '%')
            index += 1

        # f = open('test.txt', 'r', encoding='utf-8')
        # data = str(f.read()).split('\n')[:300]
        # tagger = nlp.pos(data)
        word = []
        data = str(f.read()).split('*' * 10)
        tagger = nlp.pos(data[1])

        if data[0] == '5' or data[0] == '0':
            continue

        for k in range(0, len(tagger) - 1):
            if tagger[k][0] in vocab_pos2 and tagger[k + 1][0] in vocab_pos2:
                train_x.append(list(vocab[vocab_pos2.index(tagger[k][0])]))
                train_x.append(list(vocab[vocab_pos2.index(tagger[k + 1][0])]))
                # word.append(tagger[j][0])
                # word.append(tagger[j + 1][0])

        """train_x = []
        train_y = []
        for i in range(30000, 50001):

            if i % 200 == 0:
                print(i // 200 - 150, '%')

            if os.path.exists('./../../data/movie_review/train_movie_review%d.txt' % i):
                f = open('./../../data/movie_review/train_movie_review%d.txt' % i, 'r', encoding='utf-8')
                data = str(f.readline())
                line = data.split('*'*10)
                count = 0

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

                f.close()"""

        embedding_size = 100

        if len(train_x) != 0:
            re_train_x = np.array(train_x)
            re_train_x = re_train_x.reshape(re_train_x.size // 200, 200)
            # print(re_train_x.shape)
            # train_y = np.array(train_y)
            # re_train_y = np.zeros((len(train_y), 1))
            # re_train_y = train_y[:len(train_y), :]
            # ont_hot_train_y = tf.squeeze(tf.one_hot(re_train_y, 2), axis=1)

            # batch_xs, batch_ys = next_batch(5000, train_x, ont_hot_train_y.eval())
            # correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
            # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            # print("Predictions: ", sess.run(tf.argmax(hypothesis, 1), feed_dict={X: re_train_x}))

            """correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print("Accuracy: ", sess.run(accuracy, feed_dict={X: train_x, Y: train_y}))"""
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

            if data[0] == '0':
                pre_result_negative += 1
            elif data[0] == '1':
                pre_result_positive += 1

            if pos > neg:
                next_result_positive += 1
            else:
                next_result_negative += 1

        f.close()

    print('모델 적용 전 긍정문장의 개수 :', pre_result_positive)
    print('모델 적용 전 부정문장의 개수 :', pre_result_negative)
    print('모델 적용 후 긍정문장의 개수 :', next_result_positive)
    print('모델 적용 후 부정문장의 개수 :', next_result_negative)
elif select == '2':
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

    for i in range(1, 10001):
        f = open('movie/5/movie_review%d.txt' % i, 'r', encoding='utf-8')
        train_x = []

        if i // 100 == index:
            print('progress - [' + str('#' * index) + str('-' * (100 - index) + '] ') + str(index) + '%')
            index += 1

        # f = open('test.txt', 'r', encoding='utf-8')
        # data = str(f.read()).split('\n')[:300]
        # tagger = nlp.pos(data)
        word = []
        data = str(f.read()).split('*' * 10)
        tagger = nlp.pos(data[1])

        if data[0] == '5' or data[0] == '1':
            continue

        for k in range(0, len(tagger) - 1):
            if tagger[k][0] in vocab_pos2 and tagger[k + 1][0] in vocab_pos2:
                train_x.append(list(vocab[vocab_pos2.index(tagger[k][0])]))
                train_x.append(list(vocab[vocab_pos2.index(tagger[k + 1][0])]))
                # word.append(tagger[j][0])
                # word.append(tagger[j + 1][0])

        """train_x = []
        train_y = []
        for i in range(30000, 50001):

            if i % 200 == 0:
                print(i // 200 - 150, '%')

            if os.path.exists('./../../data/movie_review/train_movie_review%d.txt' % i):
                f = open('./../../data/movie_review/train_movie_review%d.txt' % i, 'r', encoding='utf-8')
                data = str(f.readline())
                line = data.split('*'*10)
                count = 0

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

                f.close()"""

        embedding_size = 100

        if len(train_x) != 0:
            re_train_x = np.array(train_x)
            re_train_x = re_train_x.reshape(re_train_x.size // 200, 200)
            # print(re_train_x.shape)
            # train_y = np.array(train_y)
            # re_train_y = np.zeros((len(train_y), 1))
            # re_train_y = train_y[:len(train_y), :]
            # ont_hot_train_y = tf.squeeze(tf.one_hot(re_train_y, 2), axis=1)

            # batch_xs, batch_ys = next_batch(5000, train_x, ont_hot_train_y.eval())
            # correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
            # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            # print("Predictions: ", sess.run(tf.argmax(hypothesis, 1), feed_dict={X: re_train_x}))

            """correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print("Accuracy: ", sess.run(accuracy, feed_dict={X: train_x, Y: train_y}))"""
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

            if data[0] == '0':
                pre_result_negative += 1
            elif data[0] == '1':
                pre_result_positive += 1

            if pos > neg:
                next_result_positive += 1
            else:
                next_result_negative += 1

        f.close()

    print('모델 적용 전 긍정문장의 개수 :', pre_result_positive)
    print('모델 적용 전 부정문장의 개수 :', pre_result_negative)
    print('모델 적용 후 긍정문장의 개수 :', next_result_positive)
    print('모델 적용 후 부정문장의 개수 :', next_result_negative)

"""print(word)
x1 = [i for i in range(1, len(pos_ag) + 1)]
x2 = [i for i in range(1, len(neg_ag) + 1)]

plt.plot(x1, pos_ag)
plt.plot(x2, neg_ag)
plt.show()"""