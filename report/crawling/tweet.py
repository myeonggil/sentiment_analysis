# -*- coding: utf-8

try:
    import json
except ImportError:
    import simplejson as json

import tweepy
import nltk
from twitter import Twitter, OAuth, TwitterHTTPError, TwitterStream


consumer_key = 'mRe5CqLmXHmn41Jye4iAn97lX'
consumer_secret = 'LKpUNzQRtve8uT9zAF9OSTCp64TKre6cItVxAteM4IhfT9wTaF'
access_token = '745600628328562689-zIBONS7euKTIeaZ8RXoLLl5q9ktDSGN'
access_secret = '9UNMkjnfKu3yMYO9HTlE4baUj9mI4wt9eSVVHlPAtZeNl'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth)

location = "%s, %s, %s" % ("35.95", "128.25", "1000km")
keyword = "중고"

search = []
cnt = 1

results = api.home_timeline(id='echosoul1994', count=10)
f = open('twitter2.txt', 'w')
for result in results:
    print(result.text)
    f.write(result.text)
"""while cnt <= 10:
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
"""

# 새로운 방법
"""oauth = OAuth(access_token, access_secret, consumer_key, consumer_secret)

twitter_stream = TwitterStream(auth=oauth)

iterator = twitter_stream.statuses.sample()

tweet_count = 1000
for tweet in iterator:
    tweet_count -= 1
    print(json.dumps(tweet))
    if tweet_count <= 0:
        break"""

"""tweets_filenames = 'twitter.txt'
tweets_file = open(tweets_filenames, 'r')

for line in tweets_file:
    try:
        tweet = json.loads(line.strip())
        if 'text' in tweet:
            print(tweet['id'])
            print(tweet['created_at'])
            print(tweet['text'])

            print(tweet['user']['id'])
            print(tweet['user']['name'])
            print(tweet['user']['screen_name'])

            hashtags = []
            for hashtag in tweet['entities']['hashtags']:
                hashtags.append(hashtag['text'])
            print(hashtags)

    except:
        continue"""