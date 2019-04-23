import re
import pandas as pd
from time import time

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emotions = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)|\^.?\^', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emotions).replace('-', '')

    return text

df = pd.read_csv('./movie_review.csv')
stime = time()
print('전처리 시작')
df['review'] = df['review'].apply(preprocessor)
print('전처리 완료: 소요시간 [%d]초' % (time() - stime))

df.to_csv('./movie_review.csv', index=False)