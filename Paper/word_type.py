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
y_la = []
for i in range(0, len(vocab_pos2)):
    y_la.append(sum(X[i]))
x_la = [i for i in range(1, len(vocab_pos2) + 1)]

plt.scatter(x_la, y_la)
plt.show()