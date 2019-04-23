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

model_name_pos2 = './../../neg/100features_100minwords_2text_pos2'
model_pos2 = g.Doc2Vec.load(model_name_pos2)
# print(model_pos1.most_similar('슬픈'))
# print(model_pos2.most_similar('슬픈'))
# print(model_neg1.most_similar('슬픈'))
# print(model_neg2.most_similar('슬픈'))

vocab_pos = list(model_pos2.wv.vocab)
print(len(vocab_pos))
X = model_pos2[vocab_pos]

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

plt.show()

tsne = TSNE(n_components=2)
# 100개의 단어에 대해서만 시각화
X_tsne = tsne.fit_transform(X[:100, :])

df = pd.DataFrame(X_tsne, index=vocab_pos[:100], columns=['x', 'y'])
#print(df.shape)
# print(df.head(5))

fig = plt.figure()
fig.set_size_inches(40, 20)
ax = fig.add_subplot(1, 1, 1)

ax.scatter(df['x'], df['y'])

for word , pos in df.iterrows():
    ax.annotate(word, pos, fontsize=30)

plt.show()