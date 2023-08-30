import streamlit as st

from newspaper import Article # 패키지 불러오기
from newspaper import Config


user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
config = Config()
config.browser_user_agent = user_agent
# page = Article("https://www.newsweek.com/donald-trump-hillary-clinton-2020-rally-orlando-1444697", config=config)
# page.download()
# page.parse()
# st.write(page.text)

# link = 'http://cnn.com' # 뉴스 기사 URL
# link = 'https://www.investing.com/news/most-popular-news'
# link = 'https://www.yna.co.kr/theme/hotnews-history?site=hot_news_btn_more'
link = 'http://news.google.com/news?hl=ko&gl=kr&ie=UTF-8&output=rss&q=삼성전자"'
# article = Article(link, language = 'ko', config=config) # URL과 언어를 입력
article = Article(link, config=config) # URL과 언어를 입력
article.download()
article.parse()
title = article.title
text = article.text
date = article.publish_date

st.write(title)
st.write(text)
print(title)


#크롤링할 url 주소
# url = 'http://www.hani.co.kr/arti/society/health/986131.html'

#한국어이므로 language='ko'
# article = Article(url, language='ko')
# article.download()
# article.parse()
# #기사 제목 가져오기
# print(article.title)
# #기사 text
# print(article.text)
# from konlpy.tag import Kkma
# st.write(article.text)

# kkma = Kkma()
# sentences = kkma.sentences(article.text)
# print(sentences)

# st.write(sentences)


# ...
# link = 'naver뉴스 URL'

# r = requests.get(link, headers=HEADER)
# soup = BeautifulSoup(r.text, 'html.parser')
# origin_url = soup.select('#ct > div.media_end_head.go_trans > div.media_end_head_info.nv_notrans > div.media_end_head_info_datestamp > a')

# origin_url 을 출력해보면, 아래와 같이 기사 원문 URL 이 포함되어 있는걸 볼 수 있다.

# <a class="media_end_head_origin_link" data-clk="are.ori" data-extra='{"lk":{"oid":"009","aid":"eacc4e13156926cd"}}' data-gdid="009" href="https://www.mk.co.kr/article/10786834" target="_blank">기사원문</a>
# link2 = origin_url[0].attrs['href']    # 원문뉴스 URL
# article = newspaper.Article(link2)
# article.download()
# article.parse()
# ...
# [출처] newspaper3k Read timed out 문제|작성자 bycho211