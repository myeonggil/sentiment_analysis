# -*- coding: utf-8 -*-

from collections import Counter
from konlpy.tag import Twitter
from gensim.models import word2vec
import logging
# import os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO) # log 출력

num_features = 100  # 문자차원 벡터 수
min_word_count = 2 # 최소 문자 수
num_workers = 4 # 병렬 처리 스레드 수
context = 2    # 문자열 창 크기
downsampling = 1e-3 # 문자 빈도 수

nlp = Twitter()
un_word = open('./../Paper/unnecessary_word.txt', 'r', encoding='utf-8')
words = un_word.readlines()
words_list = []
for i in words:
    words_list.append(i[:-1])

review = ''
re_review = ''
token_contents = []
token_total = []
# i = 1
temp = []
# 91822 91828 120395
for i in range(1, 485521):

    f = open('./train_neg_review/train_neg_review%d.txt' % i, 'r', encoding='utf-8')

    data = f.readline()
    review = str(data).split('&')
    temp = review[:-1]

    token_contents.append(temp)


    if i % 4855 == 0:
        print('%d' % int(i/4855), '%')
    f.close()

"""while os.path.exists('./../Paper/crawling/train_neg_review/train_neg_review%d.txt' % i):
    f = open('./../Paper/crawling/train_neg_review/train_neg_review%d.txt' % i, 'r', encoding='utf-8')
    neg_review = open('./train_neg_review/train_neg_review%d.txt' % i, 'w', encoding='utf-8')

    data = f.readline()
    review = str(data)
    tagger = nlp.pos(review)

    for j in range(0, len(tagger)):

        if tagger[j][0] in words_list:
            continue

        temp.append(tagger[j][0])

    # token_contents.append(temp)
    for j in temp:
        neg_review.write(j + '&')

    temp.clear()
    tagger.clear()

    if i % 4855 == 0:
        print('%d' % int(i/4855), '%')
    i += 1
    f.close()

i = 1"""

"""while os.path.exists('./train_neg_review/train_neg_review%d.txt' % i):
    f = open('./train_neg_review/train_neg_review%d.txt' % i, 'r', encoding='utf-8')

    data = f.readline()
    review = str(data).split('&')
    review = review[:-1]

    token_contents.append(review)

    if i % 4855 == 0:
        print('%d' % int(i/4855), '%')
    i += 1
    f.close()
"""
print(len(token_contents))


model = word2vec.Word2Vec(token_contents, size=100, window=2, min_count=100,
                          workers=4, iter=100, sample=downsampling, sg=1)
print(model)
model.init_sims(replace=True)
model_name = '100features_100minwords_2text_neg2'
model.save(model_name)
token_contents.clear()