import streamlit as st
import newspaper
from newspaper import Article

from newspaper import Article # 패키지 불러오기

link = 'http://cnn.com' # 뉴스 기사 URL
article = Article(link, language = 'ko') # URL과 언어를 입력
article.download()
article.parse()
title = article.title
text = article.text
date = article.publish_date

print(text)

# cnn_paper = newspaper.build('http://cnn.com')
# url = []
# for article in cnn_paper.articles[:10]:
#     url.append(article.url)
#     articledownload()
#     parse = article.parse()
#     print(article.text[:200])
#     print(article.top_image)
#     print(article.movies)

# cnn_paper.download()
# cnn_paper.parse()
# print(cnn_paper.text)

#크롤링할 url 주소
# url = 'http://www.hani.co.kr/arti/society/health/986131.html'

# #한국어이므로 language='ko'
# article = Article(url, language='ko')
# article.download()
# article.parse()
# #기사 제목 가져오기
# print(article.title)
# #기사 text
# print(article.text)
# from konlpy.tag import Kkma

# kkma = Kkma()
# sentences = kkma.sentences(article.text)
# print(sentences)