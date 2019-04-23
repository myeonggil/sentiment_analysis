from collections import Counter
from konlpy.tag import *
from gensim.models import word2vec
import os
import warnings

warnings.filterwarnings('ignore')

# twitter 형태소 분석기 Noun, Verb, Adjective
nlp = Twitter()

token = []
word_class = ['Noun', 'Verb', 'Adjective']
# 불용어 제거


i = 1
all_f = open('./crawling/pos_review/pos_review.txt', 'a', encoding='utf-8')

while os.path.exists('./crawling/pos_review/pos_review%d.txt' % i):
    f = open('./crawling/pos_review/pos_review%d.txt' % i, 'r', encoding='utf-8')
    data = f.readline()
    review = str(data)
    review = review[7:]
    all_f.write(review)
    # tagger = nlp.pos(data)
    i += 1
    f.close()


all_f.close()