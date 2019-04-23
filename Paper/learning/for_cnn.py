import tensorflow as tf
import numpy as np
import os
from konlpy.tag import *
import gensim.models as g
import csv

"""
word2vec를 통해 학습된 데이터를 텐서플로우를 통해 
긍정과 부정으로 학습을 시킨다.......... 0을 부정 1을 긍정
train data와 test data를 이용하여 벡터값을 이용하여 
문장의 총 점의 평균을 구해 합산한 뒤에  그것들을 긍정과 부정으로 판단한다.
"""

nlp = Twitter()
model_name_pos2 = './../../neg/100features_100minwords_2text_pos2'  # 이 모델을 사용
model_pos2 = g.Doc2Vec.load(model_name_pos2)
vocab_pos2 = list(model_pos2.wv.vocab)
X = model_pos2[vocab_pos2]

print(X[0])
pos_train_index = 1
neg_train_index = 1

input_train_x = []
input_train_y = []
count = 0
max = 0

"""f1 = open('./train_result.csv', 'w', newline='')
train_csv = csv.writer(f1)
f2 = open('./test_result.csv', 'w', newline='')
test_csv = csv.writer(f2)
review_som = 0"""

"""for i in range(1, 218585):
    pos_train_review = open('./../../data/train_pos_review/train_pos_review%d.txt' % i, 'r',
                            encoding='utf-8')
    pos_train_data = str(pos_train_review.readline())
    sum += len(pos_train_data)
    if i == 218584:
        print(sum / i)

revie_sum = 0
for i in range(1, 485523):
    pos_train_review = open('./../../data/train_neg_review/train_neg_review%d.txt' % i, 'r',
                            encoding='utf-8')
    pos_train_data = str(pos_train_review.readline())
    sum += len(pos_train_data)
    if i == 485523:
        print(sum / i)

review_sum = 0
for i in range(1, 24544):

review_sum = 0
for i in range(1, 54024):
"""

"""for i in range(54594, 485523):

    pos_train_mean = 0
    neg_train_mean = 0

    if os.path.exists('./../../data/train_pos_review/train_pos_review%d.txt' % i):
        pos_train_review = open('./../../data/train_pos_review/train_pos_review%d.txt' % i, 'r',
                                encoding='utf-8')
        pos_train_data = str(pos_train_review.readline())
        result = []

        if len(pos_train_data) < 20:
            pos_train_review.close()
            continue
        pos_train_tagger = nlp.pos(pos_train_data)
        real_word = []
        for j in pos_train_tagger:
            if j[0] in vocab_pos2:
                index = vocab_pos2.index(j[0])
                pos_train_mean += np.mean(X[index])
                # result.append(np.mean(X[index]))
        # result.append(1)
        # input_train_x.append(pos_train_mean)
        # input_train_y.append(1)

        # train_csv.writerow(result)

        result_x = open('./train_result_x.txt', 'a')
        result_x.write(str(pos_train_mean) + '&')
        result_y = open('./train_result_y.txt', 'a')
        result_y.write(str(1) + '&')
        result_x.close()
        result_y.close()

    if os.path.exists('./../../data/train_neg_review/train_neg_review%d.txt' % i):
        neg_train_review = open('./../../data/train_neg_review/train_neg_review%d.txt' % i, 'r',
                                encoding='utf-8')
        neg_train_data = str(neg_train_review.readline())
        result = []
        if len(neg_train_data) < 20:
            neg_train_review.close()
            continue
        neg_train_tagger = nlp.pos(neg_train_data)
        real_word = []
        for j in neg_train_tagger:
            if j[0] in vocab_pos2:
                index = vocab_pos2.index(j[0])
                neg_train_mean += np.mean(X[index])
                # result.append(np.mean(X[index]))
        # result.append(0)
        # train_csv.writerow(result)
        # input_train_x.append(neg_train_mean)
        # input_train_y.append(0)

        result_train_x = open('./train_result_x.txt', 'a')
        result_train_x.write(str(neg_train_mean) + '&')
        result_train_y = open('./train_result_y.txt', 'a')
        result_train_y.write(str(0) + '&')
        result_train_x.close()
        result_train_y.close()
    print(i)
"""
"""for i in range(2, 54024):
    pos_test_mean = 0
    neg_test_mean = 0

    if os.path.exists('./../../data/test_pos_review/test_pos_review%d.txt' % i):
        pos_test_review = open('./../../data/test_pos_review/test_pos_review%d.txt' % i, 'r',
                               encoding='utf-8')
        pos_test_data = str(pos_test_review.readline())
        result = []
        if len(pos_test_data) < 20:
            pos_test_review.close()
            continue
        pos_test_tagger = nlp.pos(pos_test_data)
        real_word = []
        for j in pos_test_tagger:
            if j[0] in vocab_pos2:
                index = vocab_pos2.index(j[0])
                pos_test_mean += np.mean(X[index])
                # result.append(np.mean(X[index]))

        # result.append(1)

        # test_csv.writerow(result)
        # input_test_x.append(pos_test_mean)
        # input_test_y.append(1)

        result_test_x = open('./pos_test_result_x.txt', 'a')
        result_test_x.write(str(pos_test_mean) + '&')
        result_test_y = open('./pos_test_result_y.txt', 'a')
        result_test_y.write(str(1) + '&')
        result_test_x.close()
        result_test_y.close()

    if os.path.exists('./../../data/test_neg_review/test_pos_review%d.txt' % i):
        neg_test_review = open('./../../data/test_neg_review/test_pos_review%d.txt' % i, 'r',
                          encoding='utf-8')
        neg_test_data = str(neg_test_review.readline())
        result = []
        if len(neg_test_data) < 20:
            neg_test_review.close()
            continue
        neg_test_tagger = nlp.pos(pos_test_data)
        real_word = []
        for j in neg_test_tagger:
            if j[0] in vocab_pos2:
                index = vocab_pos2.index(j[0])
                neg_test_mean += np.mean(X[index])
                # result.append(np.mean(X[index]))

        # result.append(0)

        # input_test_x.append(neg_test_mean)
        # input_test_y.append(0)

        # test_csv.writerow(result)

        result_test_x = open('./neg_test_result_x.txt', 'a')
        result_test_x.write(str(neg_test_mean) + '&')
        result_test_y = open('./neg_test_result_y.txt', 'a')
        result_test_y.write(str(0) + '&')
        result_test_x.close()
        result_test_y.close()

    print(i)
"""
"""while os.path.exists('./../../data/train_pos_review/train_pos_review%d.txt' % pos_train_index) or \
        os.path.exists('./../../data/train_neg_review/train_neg_review%d.txt' % neg_train_index):

    pos_train_mean = 0
    neg_train_mean = 0

    if os.path.exists('./../../data/train_pos_review/train_pos_review%d.txt' % pos_train_index):
        pos_train_review = open('./../../data/train_pos_review/train_pos_review%d.txt' % pos_train_index, 'r', encoding='utf-8')
        pos_train_data = str(pos_train_review.readline())
        result = []

        if len(pos_train_data) < 20 or len(pos_train_data) > 40:
            pos_train_index += 1
            pos_train_review.close()
            continue
        pos_train_tagger = nlp.pos(pos_train_data)
        real_word = []
        for i in pos_train_tagger:
            if i[0] in vocab_pos2:
                index = vocab_pos2.index(i[0])
                # pos_train_mean += np.mean(X[index])
                result.append(np.mean(X[index]))
        result.append(1)
        # input_train_x.append(pos_train_mean)
        # input_train_y.append(1)

        train_csv.writerow(result)

        result_x = open('./train_result_x.txt', 'a', encoding='utf-8')
        #result_x.write(str(pos_train_mean) + '&')
        result_y = open('./train_result_y.txt', 'a', encoding='utf-8')
        #result_y.write(str(1) + '&')
        result_x.close()
        result_y.close()

        pos_train_index += 1

    if os.path.exists('./../../data/train_neg_review/train_neg_review%d.txt' % neg_train_index):
        neg_train_review = open('./../../data/train_neg_review/train_neg_review%d.txt' % neg_train_index, 'r', encoding='utf-8')
        neg_train_data = str(neg_train_review.readline())
        result = []
        if len(neg_train_data) < 20 or len(neg_train_data) > 40:
            neg_train_index += 1
            neg_train_review.close()
            continue
        neg_train_tagger = nlp.pos(pos_train_data)
        real_word = []
        for i in neg_train_tagger:
            if i[0] in vocab_pos2:
                index = vocab_pos2.index(i[0])
                # neg_train_mean += np.mean(X[index])
                result.append(np.mean(X[index]))
        result.append(0)
        train_csv.writerow(result)
        # input_train_x.append(neg_train_mean)
        # input_train_y.append(0)

        result_train_x = open('./train_result_x.txt', 'a', encoding='utf-8')
        #result_train_x.write(str(neg_train_mean) + '&')
        result_train_y = open('./train_result_y.txt', 'a', encoding='utf-8')
        #result_train_y.write(str(0) + '&')
        result_train_x.close()
        result_train_y.close()

        neg_train_index += 1

pos_test_index = 2
neg_test_index = 1
input_test_x = []
input_test_y = []
count = 1

while os.path.exists('./../../data/test_pos_review/test_pos_review%d.txt' % pos_test_index) or \
    os.path.exists('./../../data/test_neg_review/test_pos_review%d.txt' % neg_test_index):

    pos_test_mean = 0
    neg_test_mean = 0

    if os.path.exists('./../../data/test_pos_review/test_pos_review%d.txt' % pos_test_index):
        pos_test_review = open('./../../data/test_pos_review/test_pos_review%d.txt' % pos_test_index, 'r',
                                encoding='utf-8')
        pos_test_data = str(pos_test_review.readline())
        result = []
        if len(pos_test_data) < 20 or len(pos_test_data) > 40:
            pos_test_index += 1
            pos_test_review.close()
            continue
        pos_test_tagger = nlp.pos(pos_test_data)
        real_word = []
        for i in pos_test_tagger:
            if i[0] in vocab_pos2:
                index = vocab_pos2.index(i[0])
                # pos_test_mean += np.mean(X[index])
                result.append(np.mean(X[index]))

        result.append(1)

        test_csv.writerow(result)
        # input_test_x.append(pos_test_mean)
        # input_test_y.append(1)

        result_test_x = open('./test_result_x.txt', 'a', encoding='utf-8')
        result_test_x.write(str(pos_test_mean) + '&')
        result_test_y = open('./test_result_y.txt', 'a', encoding='utf-8')
        result_test_y.write(str(1) + '&')
        result_test_x.close()
        result_test_y.close()

        pos_test_index += 1

    if os.path.exists('./../../data/test_neg_review/test_pos_review%d.txt' % neg_test_index):
        neg_test_review = open('./../../data/test_neg_review/test_pos_review%d.txt' % neg_test_index, 'r',
                          encoding='utf-8')
        neg_test_data = str(neg_test_review.readline())
        result = []
        if len(neg_test_data) < 20 or len(neg_test_data) > 40:
            neg_test_index += 1
            neg_test_review.close()
            continue
        neg_test_tagger = nlp.pos(pos_test_data)
        real_word = []
        for i in neg_test_tagger:
            if i[0] in vocab_pos2:
                index = vocab_pos2.index(i[0])
                # neg_test_mean += np.mean(X[index])
                result.append(np.mean(X[index]))

        result.append(0)

        # input_test_x.append(neg_test_mean)
        # input_test_y.append(0)

        test_csv.writerow(result)

        result_test_x = open('./test_result_x.txt', 'a', encoding='utf-8')
        result_test_x.write(str(neg_test_mean) + '&')
        result_test_y = open('./test_result_y.txt', 'a', encoding='utf-8')
        result_test_y.write(str(0) + '&')
        result_test_x.close()
        result_test_y.close()

        neg_test_index += 1

    if count % 700 == 0:
        print(int(count / 700) + '%')"""