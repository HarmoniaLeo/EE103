#%%
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import nltk
from scipy import linalg

df=pd.read_csv("pacifier.csv",usecols=["review_body","star_rating"],dtype={"star_rating":np.int_})
corpus=df["review_body"].values
sr=df["star_rating"].values
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

sr-=3

key_num=len(feature_name)
train_length=200
y_train=sr[:train_length]
h_train=tfidf[:train_length]
y_test=sr[train_length:]
h_test=tfidf[train_length:]
A=np.zeros((key_num,key_num))
b=np.zeros((key_num,))
#%%
lamb=np.linspace(0.1,10,50)
rmss=[]
count=0
for l in np.nditer(lamb):
    for j in range(0,key_num):
        for i in range(0,key_num):
            if i==j:
                A[j][i]+=np.sum(h_train[...,i]*h_train[...,i])+l
            else:
                A[j][i]+=np.sum(h_train[...,i]*h_train[...,j])
        b[j]+=np.sum(h_train[...,j]*y_train)
    omega = linalg.solve(A, b)
    r_test=np.sum(h_test*omega,axis=1)
    rms=np.sqrt(np.sum((y_test-r_test)**2)/y_test.shape[0])
    rmss.append(rms)
    print(count)
    count+=1
plt.plot(lamb,rmss)
plt.show()
#%%
count=0
l=2
mat=np.zeros((5,5))
for j in range(0,key_num):
    for i in range(0,key_num):
        if i==j:
            A[j][i]+=np.sum(h_train[...,i]*h_train[...,i])+l
        else:
            A[j][i]+=np.sum(h_train[...,i]*h_train[...,j])
    b[j]+=np.sum(h_train[...,j]*y_train)
omega = linalg.solve(A, b)
r_test=np.sum(h_test*omega,axis=1)
y_test+=2
r_test+=2
r_test=np.rint(r_test)
r_test=np.where(r_test>4,4,r_test)
r_test=np.where(r_test<0,0,r_test)
for i in range(0,y_test.shape[0]):
    mat[y_test[i]][int(r_test[i])]+=1

# %%
