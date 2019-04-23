import codecs, json
from twitterscraper.query import query_tweets_once
from twitterscraper.query import query_tweets

query = '통일 since:2018-01-01 until:2018-03-31'
f = open('tongil.txt', 'a', encoding='utf-8')
i = 1

if __name__ == '__main__':
    for tweet in query_tweets(query):
        """val = 'timestamp: ' + str(tweet.timestamp) + '\n'
        f.write(val)
        val = 'likes: ' + str(tweet.likes) + '\n'
        f.write(val)
        val = 'retweetts: ' + str(tweet.retweets) + '\n'
        f.write(val)"""
        val = tweet.text + '\n'
        f.write(val)
        print(i)
        i += 1

"""for tweet in query_tweets_once(query):
    val = 'timestamp: ' + str(tweet.timestamp) + '\n'
    f.write(val)
    val = 'likes: ' + str(tweet.likes) + '\n'
    f.write(val)
    val = 'retweetts: ' + str(tweet.retweets) + '\n'
    f.write(val)
    val = tweet.text + '\n\n'
    f.write(val)
    print(i)
    i += 1"""

"""with codecs.open('tweets.json', 'r', 'utf-8') as f:
    tweets = json.load(f, encoding='utf-8')

print(tweets)"""