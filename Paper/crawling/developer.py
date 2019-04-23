# -*- coding: euc-kr -*-

import facebook
import json
from collections import Counter

obj = facebook.GraphAPI(access_token='EAACEdEose0cBAEpPtIHgYPQuD5nvu7ObAdASZBJK5LSvCafF4PP2e4aD178Arm4ZBBgAExGMuCzm5QNZBPRDkmyIfSSgRg9Skhbkh40VunH4LYeg9yt5BSG5DbHpZBJSKK3CpVkoABhhD1sZCXED8EAanJ3Jsduwq3RKS4g0tnGrVRuxp4YiUH5Lckiztd8mMOeEpHkrZAXQZDZD')

limit = int(input('검색할 feed 크기 >> '))

likes = {}
_data = {}

# me : friends 총 친구들의 수, feed 올라온 게시글, fields속성을 추가하여 사용자의 댓글을 크롤링
response = obj.get_connections(id='1797132533845332', connection_name='posts', limit=limit, fields='comments')

print(response)
_find = []
comments_size = []
data_list = {}

for i in range(0, limit):
    if 'comments' in response['data'][i]:
        if 'data' in response['data'][i]['comments']:
            # data_list.append(i)
            # print(response['data'][i]['comments']['data'])
            comments_size.append(len(response['data'][i]['comments']['data']))
            data_list[i] = len(response['data'][i]['comments']['data'])

for key, value in data_list.items():
    for j in range(0, value):
        try:
            # if userid == data["from"]["name"]:
            _data = {}
            _data["id"] = response['data'][key]['comments']['data'][j]['id']
            # _data["name"] = response['data'][key]['comments']['data']['name']
            _data["created_time"] = response['data'][key]['comments']['data'][j]['created_time']
            _data["message"] = response['data'][key]['comments']['data'][j]['message']
            _find.append(_data)
        except UnicodeEncodeError as e:
            # if userid == data["from"]["name"]:
            _data = {}
            _data["id"] = response['data'][key]['comments']['data'][j]['id']
            # _data["name"] = response['data'][key]['comments']['data']['name']
            _data["created_time"] = response['data'][key]['comments']['data'][j]['created_time']
            _data["message"] = response['data'][key]['comments']['data'][j]['message']
            _find.append(_data)
    if "paging" in response and "after" in response["paging"]:
        after = response["paging"]["cursors"]["after"]

"""def pp(o):
    print(json.dumps(o, indent=1))

pp(obj.get_object('me'))
pp(obj.get_connections('me', 'friends'))

friends = obj.get_connections("1688937971318777", "posts")['data']
"""

"""for friend in friends:
    f_like = obj.get_connections(friend['id'], "likes")['data']
    likes[friend['name']] = f_like

friends_likes = Counter([like['name'] for friend in likes for like in likes[friend] if like.get('name')])
pp(friends_likes)"""


f = open('facebook.txt', 'w', encoding='utf-8')
for data in _find:
    f.write('==' * 30 + "\n")
    f.write(str(data['created_time']) + "\n")
    f.write(str(data['message']) + "\n")
    f.write(str(data['id']) + "\n")
    # f.write(str(data["name"]) + "\n")
    f.write('==' * 30 + "\n")
f.close()