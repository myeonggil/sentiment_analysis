import pandas as pd

df = pd.DataFrame()
df = pd.read_csv('./movie_review.csv')
print(df.head())
print(df.tail())