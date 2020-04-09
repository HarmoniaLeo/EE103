import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import nltk

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
tfidf=tfidf.T
print(feature_name)
df=pd.DataFrame(tfidf,columns=["comment 1","comment 2","comment 3","comment 4","comment 5","comment 6","comment 7","comment 8","comment 9","comment 10"])
df.plot.bar()
plt.xticks([])
plt.show()