import streamlit as st
import joblib
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import os
import nltk 
import contractions
import re
from nltk.tokenize.toktok import ToktokTokenizer

nltk.download('stopwords')
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')
tokenizer = ToktokTokenizer()


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
vs = SentimentIntensityAnalyzer()

urls = ['https://inshorts.com/en/read/sports',
        'https://inshorts.com/en/read/world',
        'https://inshorts.com/en/read/politics']

def build_dataset(urls):
  news_data = []
  for url in urls:
    news_category = url.split('/')[-1]
    data = requests.get(url)
    soup = BeautifulSoup(data.content)
    
    news_articles = [{'news_headline':headline.find('span',attrs={"itemprop":"headline"}).string,
                      'news_article':article.find('div',attrs={'itemprop':'articleBody'}).string,
                      'news_category':news_category}
                     
                     for headline,article in zip(soup.find_all('div',class_=["news-card-title news-right-box"]),
                                                 soup.find_all('div',class_=["news-card-content news-right-box"]))]
    news_articles = news_articles[0:20]
    news_data.extend(news_articles)

  df = pd.DataFrame(news_data)   
  df = df[["news_headline","news_article","news_category"]]
  return df 

df = build_dataset(urls)

df.news_headline = df.news_headline.apply(lambda x:x.lower())
df.news_article = df.news_article.apply(lambda x:x.lower())

df.news_headline = df.news_headline.apply(html_tag)
df.news_article = df.news_article.apply(html_tag)

df.news_headline = df.news_headline.apply(con)
df.news_article = df.news_article.apply(con)

df.news_headline = df.news_headline.apply(remove_sp)
df.news_article = df.news_article.apply(remove_sp)

df.news_headline = df.news_headline.apply(remove_stopwords)
df.news_article = df.news_article.apply(remove_stopwords)

df['compound'] = df['news_article'].apply(lambda x: vs.polarity_scores(x)['compound'])


st.title('Sentimental Analysis')
ip = st.text_input("Enter the message")
op = vs.polarity_scores([ip])

if st.button('polarity_scores'):
  st.title(op[0])
