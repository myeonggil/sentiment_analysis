# -*- coding:utf-8 -*-

from sklearn.manifold import TSNE
import matplotlib as mpl
import matplotlib.pyplot as plt
import gensim
import gensim.models as g
import pandas as pd
from scipy.spatial import distance
import numpy as np
import csv
from datetime import datetime

# 리스트 맨 앞에 삽입 list.insert(0, x)
plt.rc('font', family='HYsanB')
plt.rcParams['axes.unicode_minus'] = False

model_name= '100features_100minwords_10text'
model = g.Doc2Vec.load(model_name)
print(model.most_similar('슬픈'))
# print(model.most_similar('행복'))

vocab_pos = list(model.wv.vocab)
print(len(vocab_pos))
X = model[vocab_pos]

# 저장 될 때는 한글이 깨져서 저장되지만 읽어올 때는 문제 없음....
# f1 = pd.read_csv('./pos_word_weight.csv', encoding='utf-8')

"""results = []
start = datetime.now()

print('거리 구하기 시작 시간: ', start)

f = open('./pos_word_weight.csv', 'w', encoding='utf-8', newline='')
vocab_pos.insert(0, '구분')
write = csv.writer(f)
write.writerow(vocab_pos)


for i in range(0, len(X)):
    weights = []
    weights.append(vocab_pos[i + 1])
    for j in range(0, len(X)):
        each_weight = round(np.exp(-((distance.euclidean(X[i], X[j])**2) / 2)), 3)
        weights.append(each_weight)
    write.writerow(weights)
    if i % 232 == 0:
        present = datetime.now()
        print('유클리디안 거리: %d' % int(i / 232) + '%', ' 시간: ', present)

# f.close()

end = datetime.now()
total = end - start
print('총 소요 시간:', total)"""

# std = np.var(results) 0에 근접하기 때문에 영향을 주지 않는다고 판단

print(np.shape(X))
