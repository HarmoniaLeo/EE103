#%%
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import nltk
from scipy.cluster.vq import kmeans,vq,whiten

k=10
corpus=pd.read_csv("pacifier.csv",usecols=["review_body"])[:100]["review_body"].values
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
tfidf=tfidf.T
feature_name = tfidf_vectorizer.get_feature_names()
data = whiten(tfidf)
centroids,_ = kmeans(data,k)
clx,_ = vq(data,centroids)
sets=[[] for i in range(0,k)]
for i in range(0,len(clx)):
    sets[clx[i]].append(feature_name[i])

# %%
