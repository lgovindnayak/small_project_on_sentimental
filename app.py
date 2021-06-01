import streamlit as st
import joblib

import pandas as pd
import numpy as np
import os
import nltk # nltk : natural process toolkit, used for remove the stop words
import contractions 
import re
from nltk.tokenize.toktok import ToktokTokenizer

nltk.download('stopwords')
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')
tokenizer = ToktokTokenizer()

df = joblib.load('sentiment')


# 1. Lower case
df.news_headline = df.news_headline.apply(lambda x:x.lower())
df.news_article = df.news_article.apply(lambda x:x.lower())

# 2. HTMP Tags
df.news_headline = df.news_headline.apply(html_tag)
df.news_article = df.news_article.apply(html_tag)

# 3. Contractions
df.news_headline = df.news_headline.apply(con)
df.news_article = df.news_article.apply(con)

# 4. Special Charcters
df.news_headline = df.news_headline.apply(remove_sp)
df.news_article = df.news_article.apply(remove_sp)

# 5. Stop Words
df.news_headline = df.news_headline.apply(remove_stopwords)
df.news_article = df.news_article.apply(remove_stopwords)

df['compound'] = df['news_article'].apply(lambda x: vs.polarity_scores(x)['compound'])


st.title('Sentimental Analysis')
ip = st.text_input("Enter the message")
op = vs.polarity_scores([ip])

if st.button('polarity_scores'):
  st.title(op[0])
