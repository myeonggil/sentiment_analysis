# -*- coding: utf-8 -*-

from collections import Counter
from konlpy.tag import Twitter
from gensim.models import word2vec
import logging
import os
import numpy as np

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
token = []
token_contents = []
temp = []
i = 1

while os.path.exists('./../Paper/crawling/train_pos_review/train_pos_review%d.txt' % i):
    f = open('./../Paper/crawling/train_pos_review/train_pos_review%d.txt' % i, 'r', encoding='utf-8')
    data = f.readline()
    review = str(data)
    tagger = nlp.pos(review)
    token.append(tagger)

    temp = []
    for j in range(0, len(token[i-1]) - 1):

        if token[i - 1][j][0] in words_list:
            continue

        temp.append(token[i - 1][j][0])

    token_contents.append(temp)

    if i % 2185 == 0:
        print('%d' % int(i/2185), '%')
    i += 1
    f.close()

print(len(token_contents))


model = word2vec.Word2Vec(token_contents, size=100, window=2, sg=1,
                          min_count=100, workers=4, iter=100, sample=downsampling)
print(model)
model.init_sims(replace=True)   # 필요없는 메모리 unload
model_name = '100features_100minwords_2text_pos2'
model.save(model_name)
