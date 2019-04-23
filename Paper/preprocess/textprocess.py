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

warnings.filterwarnings('ignore')
"""font_location = 'C:/Windows/Fonts/나눔바른고딕'
font_name = font_manager.FontProperties(fname=font_location).get_name()
rc('font', family=font_name)
"""

"""font_list = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
print(len(font_list))
print(font_list[:10])
"""

plt.rc('font', family='HYsanB')

num_feature = 300   # 문자 벡터 차원 수
min_word_count = 4  # 최소 문자 수
num_workers = 4 # cpu, 병렬 처리 스레드 수
context = 10    # 문자열 창 크기
downsampling = 1e-3 # 문자 빈도 수

nlp = Twitter()
token = []
word_class = ['Noun', 'Verb', 'Adjective']

for i in range(1, 1000):
    if os.path.exists('./crawling/movie_review/movie_review%d.txt' % i):
        f = open('./crawling/movie_review/movie_review%d.txt' % i, 'r', encoding='utf-8')
        data = f.readline()
        if data == "":
            print('공백')
            continue
        line = data.split('***********')
        if len(line[0]) > 5:
            tagger = nlp.pos(line[0])
        else:
            tagger = nlp.pos(line[1])
        for i in tagger:
            if i[1] in word_class:
                token.append(i)

print(token)
"""
# count = Counter(token)
model = word2vec.Word2Vec(token,
                          workers=num_workers,
                          size=num_feature,
                          min_count=min_word_count,
                          window=context,
                          sample=downsampling)

model.init_sims(replace=True)   # 학습이 완료되면 필요없는 메모리 unload
model_name = '300features_40windows_10text'
# model.save(model_name)

plt.rcParams['axes.unicode_minus'] = False  # 마이너스 폰트 깨지는 것 방지
model2 = g.Doc2Vec.load(model_name)

vocab = list(model2.wv.vocab)
x = model2[vocab]

print(len(x))
print(x[0][:10])
tsne = TSNE(n_components=2)

x_tsne = tsne.fit_transform(x[:100, :])
df = pd.DataFrame(x_tsne, index=vocab[:100], columns=['x', 'y'])
print(df.shape)
print(df.head(10))

fig = plt.figure()
fig.set_size_inches(100, 50)
ax = fig.add_subplot(1, 1, 1)

ax.scatter(df['x'], df['y'])

for word, pos in df.iterrows():
    ax.annotate(word, pos, fontsize=10)
plt.show()"""