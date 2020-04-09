#%%
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import nltk

def norm(arr):
    return np.sqrt(np.sum((arr)**2,axis=0))

def radian(arr1,arr2):
    return np.arccos(np.sum(arr1*arr2,axis=0)/(norm(arr1)*norm(arr2)))

def distance(arr1,arr2):
    return np.sqrt(np.sum((np.abs(arr1-arr2))**2,axis=0))

corpus=pd.read_csv("pacifier.csv",usecols=["review_body"])[:10]["review_body"].values
tokenizer = nltk.RegexpTokenizer(r'\w+')	#去除标点符号的正则过滤器
corpus2=["" for i in range(0,corpus.shape[0])]
for i in range(0,corpus.shape[0]):
    lis=tokenizer.tokenize(corpus[i])
    for word in lis:
        corpus2[i]+= nltk.PorterStemmer().stem(word)+" "
        #将文本中所有单词只保留词干
corpus=np.array(corpus2)
tfidf_vectorizer = TfidfVectorizer() 
tfidf = tfidf_vectorizer.fit_transform(corpus).toarray()
feature_name = tfidf_vectorizer.get_feature_names()
sm_dis=np.zeros((tfidf.shape[0],tfidf.shape[0]))
sm_rad=np.zeros((tfidf.shape[0],tfidf.shape[0]))
for j in range(0,tfidf.shape[0]):
    for i in range(j,tfidf.shape[0]):
        sm_dis[j][i]=distance(tfidf[j],tfidf[i])
        sm_dis[i][j]=sm_dis[j][i]
        sm_rad[j][i]=radian(tfidf[j],tfidf[i])
        sm_rad[i][j]=sm_rad[j][i]

# %%
