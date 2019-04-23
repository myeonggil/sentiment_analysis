from bs4 import BeautifulSoup
import requests
from threading import Thread

def crawling_data(start, end):

    title_index = 0
    score_index = 1 + ((start-1) * 15)
    review_index = 1 + ((start-1) * 15)
    dif = int((end - start)/100)
    compare = (start - 1) + dif

    for index in range(start, end):
        # url = 'http://movie.naver.com/movie/point/af/list.nhn?&page=%d' % i # 네티즌 평점
        url = 'http://movie.naver.com/movie/board/review/list.nhn?&page=%d' % index  # 네티즌 리뷰
        # url = 'http://movie.naver.com/movie/board/review/'
        html = get_html(url)
        soup = BeautifulSoup(html, 'html.parser')

        scores = soup.find_all('div', {'class': 'mask'})
        links = soup.find_all('td', {'class': 'title'})

        for score in scores:
            f = open('./data/train_movie_review%d.txt' % score_index, 'w', encoding='utf-8')
            str_score = str(score)
            start_score_index = str_score.find('alt=\"')
            end_score_index = str_score.find('점')
            # re_score = str_score[start_score_index + 5:end_score_index] + '*' * 10
            re_score = int(str_score[start_score_index + 5:end_score_index])
            if re_score >= 7:
                re_score = '1' + '*'*10
            elif re_score >= 1 and re_score <= 4 :
                re_score = '0' + '*'*10
            else:
                re_score = '5' + '*'*10

            f.write(re_score)
            score_index += 1
            f.close()

        for link in links:
            f = open('./data/train_movie_review%d.txt' % review_index, 'a', encoding='utf-8')
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
