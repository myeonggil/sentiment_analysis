from bs4 import BeautifulSoup
import requests

review1 = 1
review2 = 1
def get_html(url):
    _html = ""
    resp = requests.get(url)
    if resp.status_code == 200:
        _html = resp.text

    return _html

for i in range(1, 180):
    url = 'http://movie.naver.com/movie/bi/mi/review.nhn?code=85579&page=%d' % i
    html = get_html(url)
    soup = BeautifulSoup(html, 'html.parser')
    my_title1 = soup.select('p > a')
    my_title1 = my_title1[7:15]
    my_title2 = soup.select('ul > li > a > strong')
    my_title2 = my_title2[8:17]
    for title in my_title1:
        f = open('C:/Users/User/PycharmProjects/Paper/crawling/Along_with_the_God_review/review1_%d.txt' % review1, 'w', encoding='utf-8')
        f.write(title.text)
        f.close()
        review1 += 1

    for title in my_title2:
        f = open('C:/Users/User/PycharmProjects/Paper/crawling/Along_with_the_God_review/review2_%d.txt' % review2, 'w', encoding='utf-8')
        f.write(title.text)
        f.close()
        review2 += 1

"""url = 'http://movie.naver.com/movie/bi/mi/review.nhn?code=85579&page=1'
html = get_html(url)
soup = BeautifulSoup(html, 'html.parser')
my_title1 = soup.select('p > a')
my_title1 = my_title1[7:15]
my_title2 = soup.select('ul > li > a > strong')
my_title2 = my_title2[8:17]
for title in my_title1:
    print(title.text)

print("\n")

for title in my_title2:
    print(title.text)

f = open('C:/Users/User/PycharmProjects/Paper/crawling/Along_with_the_God_review/test.txt', 'w', encoding='utf-8')
f.write("asd")
f.close()"""