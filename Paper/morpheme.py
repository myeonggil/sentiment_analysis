import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import gensim.models as g
from konlpy.tag import *
import numpy as np


model_name_pos2 = './../neg/100features_100minwords_2text_pos2'
model_pos2 = g.Doc2Vec.load(model_name_pos2)

vocab_pos2 = list(model_pos2.wv.vocab)
X = model_pos2[vocab_pos2]
train_x = []
train_y = []
pos = 0
neg = 0
max = 20
nlp = Twitter()
pos_mean = []
neg_mean = []
x1 = []
x2 = []
index = 0
plot = []

for i in range(20000, 40001):
    mean = 0
    if i % 400 == 0:
        print(i // 400, '%')

    if os.path.exists('./../data/movie_review/train_movie_review%d.txt' % i):
        f = open('./../data/movie_review/train_movie_review%d.txt' % i, 'r', encoding='utf-8')
        data = str(f.readline())
        line = data.split('*'*10)
        count = 0

        if line[0] == '1':
            pos += 1
            if pos >= max:
                continue

        if line[0] == '0':
            neg += 1
            if neg >= max:
                continue

        if line[0] != '5' and line[0] == '1':

            tagger = nlp.pos(line[1])
            for j in range(0, len(tagger) - 1):
                if tagger[j][0] in vocab_pos2 and tagger[j + 1][0] in vocab_pos2:
                    count += 1


            if count != 0:
                for j in range(0, len(tagger)):
                    if tagger[j][0] in vocab_pos2:
                        # pos_mean.append(np.mean(X[vocab_pos2.index(tagger[j][0])]))
                        mean += np.mean(X[vocab_pos2.index(tagger[j][0])])
            pos_mean.append(mean + 0.1)

        if line[0] != '5' and line[0] == '0':

            tagger = nlp.pos(line[1])
            for j in range(0, len(tagger) - 1):
                if tagger[j][0] in vocab_pos2 and tagger[j + 1][0] in vocab_pos2:
                    count += 1


            if count != 0:
                for j in range(0, len(tagger)):
                    if tagger[j][0] in vocab_pos2:
                        # neg_mean.append(np.mean(X[vocab_pos2.index(tagger[j][0])]))
                        mean += np.mean(X[vocab_pos2.index(tagger[j][0])])
            neg_mean.append(mean - 0.1)

        f.close()

index = 1
x = []
print(len(pos_mean), len(neg_mean))
for i in range(0, 18):
    before = (pos_mean[i] + neg_mean[i]) / 2
    after = (pos_mean[i + 1] + neg_mean[i + 1]) / 2
    dif = (after - before) / 1000
    plot.append(before)
    for j in range(1000):
        plot.append(before + dif)

plot.append((pos_mean[18] + neg_mean[18]) / 2)

for i in range(0, 18):
    x.append(i + 1)
    dif = (i + 1) / 1000
    for j in range(1000):
        x.append((i + 1) + dif)

x.append(18)

for i in range(0, len(pos_mean)):
    x1.append(i + 1)

for i in range(0, len(neg_mean)):
    x2.append(i + 1)


fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.plot(x, plot, label='median')
ax.scatter(x1, pos_mean, c='r', label='Positive', marker='*')
ax.scatter(x2, neg_mean, c='b', label='Negative')
ax.legend(loc='upper left')
plt.show()