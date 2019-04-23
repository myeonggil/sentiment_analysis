# -*- coding: utf-8 -*-

import facebook
import json
from collections import Counter

obj = facebook.GraphAPI(access_token='EAACEdEose0cBAJXRbFxZCVCOOfdqLLuHQleWp4eCR3gEYTg01ZAkEJtxX1ZCFOx5O1Ca6Lmxx4IWKDfpjjkfVHWbudwBUZCRNPXosa4towxJZAZAJXlrvlByTMj1brUZCdv3S6OWhEqZAbVuTbZBipYZBdeUJFexV9cjFZBUzevfmK7zLZAOpZBZB4JV6ZBldoTqa8DXb4ZD')

"""def pp(o):
    print(json.dumps(o, indent=1))

pp(obj.get_object('me'))
pp(obj.get_connections('me', 'friends'))"""

#friends = obj.get_connections("1688937971318777", "posts")['data']

likes = {}
"""for friend in friends:
    f_like = obj.get_connections(friend['id'], "likes")['data']
    likes[friend['name']] = f_like

friends_likes = Counter([like['name'] for friend in likes for like in likes[friend] if like.get('name')])
pp(friends_likes)"""

#postid = str(input("포스트 아이디 입력 : "))
#Auserid = input("찾을 사용자 아이디 입력 : ")
# me : friends 총 친구들의 수, feed 올라온 게시글, fields속성을 추가하여 사용자의 댓글을 크롤링
response = obj.get_connections(id='1453931941512003', connection_name='feed', limit=40, fields='comments')

print(response['data'])

_find = []
comment_size = []
message_size = []

feed_size = len(response['data'])
for i in range(0, feed_size):
    comment_size.append(len(response['data'][i]))
    print(comment_size)

for i in range(0, feed_size):
    print(response['data'][i])
    print(response['data'][i]['comments']['data'])

"""for i in range(0, feed_size):
    for j in range(0, comment_size[i]):
        print(response['data'][i]['comments']['data'])"""

"""for i in range(0, feed_size):
    for j in range(0, comment_size[i]):
        message_size.append(len(response['data'][j]['comments']['data']))
        print(message_size)"""

for i in range(0, feed_size):
    for j in range(0, comment_size[i]):
        for k in range(0, message_size[j]):
            try:
                # if userid == data["from"]["name"]:
                _data = {}
                _data["id"] = response["data"][i]['comments']['data'][j]["from"]["id"][k]
                _data["name"] = response["data"][i]['comments']['data'][j]["from"]["name"][k]
                _data["created_time"] = response["data"][i]['comments']['data'][j]["created_time"]
                _data["message"] = response["data"][i]['comments']['data'][j]["message"]
                _find.append(_data)
                print(_data)
            except UnicodeEncodeError as e:
                # if userid == data["from"]["name"]:
                _data = {}
                _data["id"] = response["data"][i]['comments']['data'][j]["from"]["id"]
                _data["name"] = response["data"][i]['comments']['data'][j]["from"]["name"]
                _data["created_time"] = response["data"][i]['comments']['data'][j]["created_time"]
                _data["message"] = response["data"][i]['comments']['data'][j]["message"]
                _find.append(_data)

if "paging" in response and "after" in response["paging"]:
    after = response["paging"]["cursors"]["after"]
    response = obj.get_connections(id='me', connection_name="comments", limit=40, after=after)


"""with response['data'][index]['comments']['data'] as data:
    while(index < feed_size):
        try:
            # if userid == data["from"]["name"]:
            _data = {}
            _data["id"] = data["from"]["id"]
            _data["name"] = data["from"]["name"]
            _data["created_time"] = data["created_time"]
            _data["message"] = data["message"]
            _find.append(_data)
        except UnicodeEncodeError as e:
            # if userid == data["from"]["name"]:
            _data = {}
            _data["id"] = data["from"]["id"]
            _data["name"] = data["from"]["name"]
            _data["created_time"] = data["created_time"]
            _data["message"] = data["message"]
            _find.append(_data)
    if "paging" in response and "after" in response["paging"]:
        after = response["paging"]["cursors"]["after"]
        response = obj.get_connections(id='me', connection_name="comments", limit=40, after=after)
        index += 1
"""


while response["data"]:
    for data in response["data"][0]['comments']['data']:
        try:
             #if userid == data["from"]["name"]:
             _data = {}
             _data["id"] = data["from"]["id"]
             _data["name"] = data["from"]["name"]
             _data["created_time"] = data["created_time"]
             _data["message"] = data["message"]
             _find.append(_data)
        except UnicodeEncodeError as e:
            #if userid == data["from"]["name"]:
             _data = {}
             _data["id"] = data["from"]["id"]
             _data["name"] = data["from"]["name"]
             _data["created_time"] = data["created_time"]
             _data["message"] = data["message"]
             _find.append(_data)
    if "paging" in response and "after" in response["paging"]:
        after = response["paging"]["cursors"]["after"]
        response = obj.get_connections(id='me', connection_name="comments", limit=40, after=after)
    else:
        break

f = open("facebook.txt", "w")
for data in _find:
    f.write("==" * 30 + "\n")
    f.write(str(data["created_time"]) + "\n")
    f.write(str(data["message"]) + "\n")
    f.write(str(data["id"]) + "\n")
    f.write(str(data["name"]) + "\n")
    f.write("==" * 30 + "\n")
f.close()