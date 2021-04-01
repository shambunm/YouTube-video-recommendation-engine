# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 19:26:59 2021

@author: shambhu
"""
import pandas as pd
import streamlit as st 

st.sidebar.header('User Input Parameters')
topN = st.sidebar.number_input("Top N")

import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

df = pd.read_csv('../project/videoid.csv', index_col=0)
pd.set_option('display.max_columns', None)
# df.shape
df.fillna('AI', inplace=True)
# df.isnull().sum()
# df.tail()

st.title('YouTube Recomendation Engine')
st.header('Select YouTube Video')
video_id_input = st.selectbox('ID / Title',(df.videoid))

query_index = 0
for i in range(0, len(df.videoid)):
    if (video_id_input == df.videoid[i]):
        query_index = i
        break

url_1 = "https://www.youtube.com/watch?v=" + df.videoid[query_index]
st.write(df.loc[query_index,"title"][:80])
st.video(url_1)


views = df['views']
like = df['like']
dislike = df['dislike']
df['rating'] = (views * (( like - dislike )/ (like + dislike)))

df1 = df.loc[:,('videoid', 'like', 'dislike','views','rating', "description", 'title')]
df1.isnull().sum()
df1.fillna(0, inplace = True)

i = df1['rating']
df1['rating'] = (i-i.min())/ (i.max()-i.min())*100
# df1.head()
# df1.describe()

wl =WordNetLemmatizer()
title1= []
for review in df1.description:
    review = review.split()
    clean_review = []
    for word in review:
        clean_review.append(wl.lemmatize(word.lower()))
        # clean_review.append(word.lower())
    clean_review = " ".join(clean_review)
    title1.append(clean_review)
# len(title1)

title2 = []
for review in title1:
    title2.append(re.sub("[^a-zA-Z0-9]+",' ', review))
# len(title2)

df1['title2'] = pd.DataFrame(title2)
# df1.tail()
# df1.describe()


from sklearn.feature_extraction.text import TfidfVectorizer 
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df1.title2)
# tfidf_matrix.shape 
# type(tfidf_matrix)

# from sklearn.metrics.pairwise import linear_kernel
# cosine_sim_matrix = linear_kernel(tfidf_matrix,tfidf_matrix)
# cosine_sim_matrix[2]
# cosine_sim_matrix.shape


from sklearn.metrics.pairwise import cosine_similarity
cosine_sim_matrix = cosine_similarity(tfidf_matrix,tfidf_matrix)
# cosine_sim_matrix[2][150]
# cosine_sim_matrix.shape

df_index = pd.Series(df1.index,index=df1['videoid'])
# df_index.head(10)
# df_index.shape
# df_index["ad79nYk2keg"]
# cosine_scores = list(enumerate(cosine_sim_matrix[6]))


def recommendations(Id,topN):
    
    video_id = df_index[Id]
    cosine_scores = list(enumerate(cosine_sim_matrix[video_id]))
    cosine_scores = sorted(cosine_scores,key=lambda x:x[1],reverse = True)
    cosine_scores_10 = cosine_scores[1:topN]

    video_idx  =  [i[0] for i in cosine_scores_10]
    video_scores =  [i[1] for i in cosine_scores_10]
    
    video_similar = pd.DataFrame(columns=["videoid","views","Score", "rating",'title'])
    video_similar["videoid"] = df1.loc[video_idx,"videoid"]
    video_similar["views"] = df1.loc[video_idx,"views"]
    video_similar["rating"] = df1.loc[video_idx,"rating"]
    video_similar["title"] = df1.loc[video_idx,"title"]
    video_similar["Score"] = video_scores
    video_similar.reset_index(inplace=True)  
    video_similar.drop(["index"],axis=1,inplace=True)
    return pd.DataFrame(video_similar)

result = recommendations(video_id_input,int(topN))
result = result.sort_values(by = 'rating', ascending = False) 
model = pd.DataFrame(result.head(10))
model.reset_index(inplace=True)

index = pd.Series(model.index,index=model['videoid'])

st.sidebar.header('Predicted Result')
for j in model.videoid:
    url = "https://www.youtube.com/watch?v="+j
    st.sidebar.write(model.loc[index[j],"title"][:80])
    st.sidebar.video(url)