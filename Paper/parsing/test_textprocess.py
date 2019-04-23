import os
from konlpy.tag import *
from gensim.models import word2vec
import warnings
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import gensim
import gensim.models as g
from collections import Counter
import pandas as pd
from matplotlib import rc, font_manager




nlp = Twitter()
token = []

pos = '1'
neg = '0'
pos_index = 1
neg_index = 1
per = 1

for i in range(1, 115000):
    if os.path.exists('./crawling/test_movie_review/movie_review%d.txt' % i):
        f = open('./crawling/test_movie_review/movie_review%d.txt' % i, 'r', encoding='utf-8')
        data = f.readline()
        line = data.split('**********')

        if len(line) == 2:
            if len(line[0]) > 5:
                review_data = line[0][7:]
                dif = line[1]
            elif len(line[1]) > 5:
                review_data = line[1][7:]
                dif = line[0]
            else:
                f.close()
                continue

            if dif == pos:
                pos_review = open('./crawling/test_pos_review/test_pos_review%d.txt' % pos_index, 'w', encoding='utf-8')
                pos_review.write(review_data)
                pos_review.close()
                pos_index += 1
            else:
                neg_reivew = open('./crawling/test_neg_review/test_pos_review%d.txt' % neg_index, 'w', encoding='utf-8')
                neg_reivew.write(review_data)
                neg_reivew.close()
                neg_index += 1
            f.close()

    if per == int(i / 1150):
        print('%d' % per, '% success')
        per += 1