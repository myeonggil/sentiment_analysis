import csv
from bs4 import BeautifulSoup
import requests
from konlpy.tag import *
import nltk
from threading import Thread

review_index = 1
score_index = 1
title_index = 0

def get_html(url):
    _html = ''
    resp = requests.get(url)
    if resp.status_code == 200:
        _html = resp.text

    return _html

for index in range(1, 10001):
    # url = 'http://movie.naver.com/movie/point/af/list.nhn?&page=%d' % i # 네티즌 평점
    url = 'http://movie.naver.com/movie/board/review/list.nhn?&page=%d' % index  # 네티즌 리뷰
    # url = 'http://movie.naver.com/movie/board/review/'
    html = get_html(url)
    soup = BeautifulSoup(html, 'html.parser')

    scores = soup.find_all('div', {'class': 'mask'})
    links = soup.find_all('td', {'class': 'title'})

    for score in scores:
        f = open('./crawling/test_movie_review/movie_review%d.txt' % score_index, 'a', encoding='utf-8')
        str_score = str(score)
        start_score_index = str_score.find('alt=\"')
        end_score_index = str_score.find('점')
        # re_score = str_score[start_score_index + 5:end_score_index] + '*' * 10
        re_score = int(str_score[start_score_index + 5:end_score_index])
        if re_score >= 6:
            re_score = '1' + '*'*10
        else:
            re_score = '0' + '*'*10
        f.write(re_score)
        f.close()
        score_index += 1

    for link in links:
        f = open('./crawling/test_movie_review/movie_review%d.txt' % review_index, 'a', encoding='utf-8')
        str_link = str(link)
        start_link_index = str_link.find('href=\"')
        end_link_index = str_link.rfind('\">')
        re_link = str_link[start_link_index + 6:end_link_index].split('amp;')
        url = 'http://movie.naver.com/movie/board/review/'
        for j in re_link:
            url += j

        html = get_html(url)
        soup = BeautifulSoup(html, 'html.parser')
        my_title = soup.select('p')
        title_len = len(my_title)
        for title in my_title:
            if title_index < title_len - 11:
                f.write(title.text)
            title_index += 1
        f.close()
        review_index += 1
        title_index = 0
