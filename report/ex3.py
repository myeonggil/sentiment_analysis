from collections import Counter
from konlpy.tag import Twitter
import pytagcloud
import nltk

f = open("test.txt")
data = f.read()
texts = nltk.word_tokenize(data)
nlp = Twitter()
# print(nlp.nouns(data))
# print(nltk.pos_tag(texts))

result = nltk.pos_tag(texts)
for i, j in enumerate(nltk.pos_tag(texts)):
    print(j[0])