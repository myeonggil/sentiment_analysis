import tweepy
from tweepy import OAuthHandler
import json
import wget

consumer_key = 'mRe5CqLmXHmn41Jye4iAn97lX'
consumer_secret = 'LKpUNzQRtve8uT9zAF9OSTCp64TKre6cItVxAteM4IhfT9wTaF'
access_token = '745600628328562689-zIBONS7euKTIeaZ8RXoLLl5q9ktDSGN'
access_secret = '9UNMkjnfKu3yMYO9HTlE4baUj9mI4wt9eSVVHlPAtZeNl'

@classmethod
def parse(cls, api, raw):
    status = cls.first_parse(api, raw)
    setattr(status, 'json', json.dumps(raw))
    return status

tweepy.models.Status.first_parse = tweepy.models.Status.parse
tweepy.models.Status.parse = parse
tweepy.models.User.first_parse = tweepy.models.User.parse
tweepy.models.User.parse = parse

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth)

tweets = api.user_timeline(screen_name='대한항공',
                           count=200, include_rts=False,
                           exclude_replies=True)

last_id = tweets[-1].id
username = 'myeonggil'
while True:
    more_tweets = api.user_timeline(screen_name=username,
                                    count=200,
                                    include_rts=False,
                                    exclude_replies=True,
                                    max_id=last_id - 1)
    if len(more_tweets) == 0:
        break
    else:
        last_id = more_tweets[-1].id - 1
        tweets = tweets + more_tweets

media_files = set()
for status in tweets:
    media = status.entities.get('media', [])
    if len(media) > 0:
        media_files.add(media[0]['media_url'])

for media_file in media_files:
    wget.download(media_file)