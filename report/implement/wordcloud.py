from collections import Counter
from konlpy.tag import Twitter
import pytagcloud
import nltk

file_name = input("파일명 입력: ")
num = int(input("파일 순번(중복x): "))
f = open(file_name)
data = f.read()
nlp = Twitter()
nouns1 = nlp.nouns(data)
nouns2 = nltk.pos_tag(data)
count = Counter(nouns1)

tag2 = count.most_common(40)
taglist = pytagcloud.make_tags(tag2, maxsize=80)

pytagcloud.create_tag_image(taglist, 'wordcloud'+ num +'.jpg', size=(900, 600), fontname='korean', rectangular=False)