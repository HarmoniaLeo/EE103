import numpy as np
import random
import matplotlib.pyplot as plt
from lagb import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd

T=200	#总时长

def regression(x,M,stop):    #M为记忆时长，stop为训练停止点
    y=pd.Series(x[M+1:stop])
    data=[]
    for i in range(0,stop-M-1):
        data.append(x[i:i+M])
    data=np.stack(data,axis=0)
    x=pd.DataFrame(data)
    linreg = LinearRegression()
    linreg.fit(x, y)
    return linreg.coef_,linreg.intercept_

x=np.empty((T,))
for i in range(0,T):
    x[i]=random.randrange(-20,20)+i
inter,c=regression(x,50,150)
x1=np.empty(x.shape)
x1[:150]=x[:150]
for i in range(150,200):
    x1[i]=np.sum(inter*x[i-50:i])+c
x2 = np.arange(0,T)
plt.plot(x2,x1,label='predict',color='y')
plt.plot(x2,x,label='original')	 
plt.legend()
plt.show()