from bs4 import BeautifulSoup
import requests

score_index = 1
review_index = 1

def get_html(url):
    _html = ""
    resp = requests.get(url)
    if resp.status_code == 200:
        _html = resp.text

    return _html

per = 1

for index in range(1, 1001):

    if index // 10 == per:
        print('progress - [' + str('#' * per) + str('-' * (100 - per) + '] ') + str(per) + '%')
        per += 1

    url = 'https://movie.naver.com/movie/point/af/list.nhn?st=mcode&sword=146506&target=&page=%d' % index
    html = get_html(url)
    soup = BeautifulSoup(html, 'html.parser')

    scores = soup.find_all('td', {'class': 'point'})
    links = soup.find_all('td', {'class': 'title'})

    for score in scores:
        f = open('5/movie_review%d.txt' % score_index, 'w', encoding='utf-8')
        str_score = str(score)
        start_score_index = str_score.find('\">')
        end_score_index = str_score.find('</')
        re_score = int(str_score[start_score_index + 2: end_score_index])
        if re_score >= 7:
            re_score = '1' + '*' * 10
        elif re_score >= 1 and re_score <= 4:
            re_score = '0' + '*' * 10
        else:
            re_score = '5' + '*' * 10

        f.write(re_score)
        score_index += 1
        f.close()

    for link in links:
        f = open('5/movie_review%d.txt' % review_index, 'a', encoding='utf-8')
        str_link = str(link)
        start_link_index = str_link.find('br/>')
        end_link_index = str_link.rfind('<a class=\"report\"')
        re_link = str_link[start_link_index + 4: end_link_index - 40]

        f.write(re_link)
        review_index += 1
        f.close()