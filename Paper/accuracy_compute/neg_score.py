import numpy as np
from scipy import interpolate
import pylab as py
from konlpy.tag import Twitter
import csv


def softmax_func(x):
    return 1 / (1 + np.exp(-x))

nlp = Twitter()

results = 0

for i in range(4, 5):
    # pos_review = open('./../../data/test_neg_review/test_pos_review%d.txt' % i, 'r', encoding='utf-8')
    pos_review = open('./ex.txt', 'r', encoding='utf-8')
    pos_line = str(pos_review.readline())
    pos_tagger = nlp.pos(pos_line)

    tagger = []
    re_tagger = []
    neg_weight_index = []

    for key in pos_tagger:
        tagger.append(key[0])

    if len(tagger) <= 1:
        continue

    neg_score = 0
    count = 1
    has = True

    neg_csv = open('./../../data/neg_word_weight.csv', 'r', encoding='utf-8')

    neg_data = csv.reader(neg_csv)

    count = 1
    has = True

    # 부정문
    for x in neg_data:
        if has:
            has = False
            for y in tagger:
                if y in x:
                    re_tagger.append(y)
                    neg_weight_index.append(x.index(y))

        elif x[0] in re_tagger:
            if x[0] != re_tagger[-1]:
                find = re_tagger.index(x[0])
                neg_score += float(x[neg_weight_index[find + 1]])

        if count % 290 == 0:
            print('neg : %d' % int(count / 290), '%')
        count += 1

    print('neg review score : ', neg_score)

    if i % 2186 == 0:
        print('accuracy percent: %d' % int(i/2186), '%')

    neg_csv.close()



"""print('긍정 : ', pos_score)
print('부정 : ', neg_score)

print('긍정 : ', softmax_func(pos_score))
print('부정 : ', softmax_func(neg_score))

def func(x):
    return x*np.exp(-5.0*x**2)

x = np.random.uniform(-1.0, 1.0, size=10)
fvals = func(x)
py.figure(1)
py.clf()
# py.plot(x, fvals, 'ro')
xnew = np.linspace(-1, 1, 100)

for kind in ['gaussian']:
    newfunc = interpolate.Rbf(x, fvals, function=kind)
    fnew = newfunc(xnew)
    py.plot(xnew, fnew, label=kind)

py.plot(xnew, func(xnew), label='true')
py.legend(loc='lower right')
py.show()"""