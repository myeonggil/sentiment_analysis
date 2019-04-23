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
un_word = open('./unnecessary_word.txt', 'r', encoding='utf-8')
words = un_word.readlines()
words_list = []
for i in words:
    words_list.append(i[:-1])

review = ''
token_contents = []
token_total = []
i = 1
temp = []

while os.path.exists('./crawling/neg_review/neg_review%d.txt' % i):
    f = open('./crawling/neg_review/neg_review%d.txt' % i, 'r', encoding='utf-8')
    print(i)
    data = f.readline()
    review = str(data)
    review = review[7:]
    tagger = nlp.pos(review)

    for j in range(0, len(tagger)):

        if tagger[j][0] in words_list:
            continue

        temp.append(tagger[j][0])

    token_contents.append(temp)
    temp.clear()
    tagger.clear()

    if i % 3770 == 0:
        token_total += token_contents
        print('%d' % int(i/3770), '%')
        token_contents.clear()
    i += 1
    f.close()

print(len(token_total))


model = word2vec.Word2Vec(token_total, size=100, window=2, min_count=300, workers=4, iter=100, sample=downsampling)
print(model)
model.init_sims(replace=True)
model_name = '100features_40minwords_10text_neg'
model.save(model_name)
