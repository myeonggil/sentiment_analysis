from bs4 import BeautifulSoup
import requests
from threading import Thread


def get_html(url):
    _html = ""
    resp = requests.get(url)
    if resp.status_code == 200:
        _html = resp.text

    return _html

url = 'http://movie.naver.com/movie/point/af/list.nhn?&page=1002'  # 네티즌 평점
# url = 'http://movie.naver.com/movie/board/review/list.nhn?&page=%d' % index  # 네티즌 리뷰
# url = 'http://movie.naver.com/movie/board/review/'
html = get_html(url)
soup = BeautifulSoup(html, 'html.parser')

scores = soup.find_all('td', {'class': 'point'})
links = soup.find_all('td', {'class': 'title'})

print(scores)
print(links)
# score = soup.select('div < div < div < div < div < table < tbody < tr < td')

"""def crawling_data(start, end):

    title_index = 0
    score_index = 1 + ((start-1) * 15)
    review_index = 1 + ((start-1) * 15)
    dif = int((end - start)/100)
    compare = (start - 1) + dif

    for index in range(start, end):
        url = 'http://movie.naver.com/movie/point/af/list.nhn?&page=%d' % i # 네티즌 평점
        # url = 'http://movie.naver.com/movie/board/review/list.nhn?&page=%d' % index  # 네티즌 리뷰
        # url = 'http://movie.naver.com/movie/board/review/'
        html = get_html(url)
        soup = BeautifulSoup(html, 'html.parser')

        scores = soup.find_all('div', {'class': 'mask'})
        links = soup.find_all('td', {'class': 'title'})

        for score in scores:


        for link in links:


        if index == compare:
            progress = (index - start) // dif
            percent_pos = '#'*progress
            percent_neg = '-'*(dif-progress)
            print(start, ':', 'progressbar[', percent_pos, percent_neg, ']', progress, '%')
            compare += dif

def get_html(url):
    _html = ""
    resp = requests.get(url)
    if resp.status_code == 200:
        _html = resp.text

    return _html

start = 1
end = 100001
threads = int(end / 20000) + 1
re_start = start
re_end = 20001

for i in range(start, threads):
    thread = Thread(target=crawling_data, args=(re_start, re_end))
    thread.start()
    re_start += 20000
    re_end += 20000
"""