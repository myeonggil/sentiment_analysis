# -*- coding: utf-8

import tweepy
import time

consumer_key = 'mRe5CqLmXHmn41Jye4iAn97lX'
consumer_secret = 'LKpUNzQRtve8uT9zAF9OSTCp64TKre6cItVxAteM4IhfT9wTaF'
access_token = '745600628328562689-zIBONS7euKTIeaZ8RXoLLl5q9ktDSGN'
access_secret = '9UNMkjnfKu3yMYO9HTlE4baUj9mI4wt9eSVVHlPAtZeNl'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth)

location = "%s, %s, %s" % ("35.95", "128.25", "1000km")
keyword = "국정농단"

search = []
cnt = 1
while cnt <= 10:
    tweets = api.search(keyword)
    for tweet in tweets:
        search.append(tweet)
    cnt += 1

data = {}
i = 0
for tweet in search:
    data['text'] = tweet.text
    print(i, " : ", data)
    i += 1

"""for i in range(1, 10):
    api.update_status('테스트. {}번째'.format(i))
    time.sleep(2)

for status in tweepy.Cursor(api.home_timeline).items():
    print(status.text)
    time.sleep(0.3)

for status in tweepy.Cursor(api.home_timeline).items():
    print(status._json)"""