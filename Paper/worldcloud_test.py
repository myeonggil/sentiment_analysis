from collections import Counter
from konlpy.tag import *
import pytagcloud

f = open('test.txt', 'r', encoding='utf-8')
data = f.read()
# nlp = Twitter()
kkma = Kkma()
nouns3 = kkma.pos(data)
nng_list = []

for i in nouns3:
    if i[1] == 'NNG':
        nng_list.append(i[0])

count = Counter(nng_list)


tag2 = count.most_common(40)
taglist = pytagcloud.make_tags(tag2, maxsize=80)

pytagcloud.create_tag_image(taglist, 'cloud.jpg', size=(900, 600), fontname='korean', rectangular=False)