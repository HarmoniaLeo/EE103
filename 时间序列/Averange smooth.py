import numpy as np
import random
import matplotlib.pyplot as plt
from lagb import *

T=200	#总时长
P=10	#周期长度
n=int(T/P)	#周期个数

def smooth(x,k):
    if k<=1:
        return x
    A=np.zeros((x.shape[0]-k,x.shape[0]))
    for i in range(0,x.shape[0]-k):
        for j in range(0,k):
            A[i][i+j]=1/int(k)
    return dot(A,x)

x=np.empty((T,))
for i in range(0,n):
    x[i*10]=random.randrange(5,10)
    x[i*10+1]=random.randrange(10,15)
    x[i*10+2]=random.randrange(15,20)
    x[i*10+3]=random.randrange(20,25)
    x[i*10+4]=random.randrange(25,30)
    x[i*10+5]=random.randrange(20,25)
    x[i*10+6]=random.randrange(15,20)
    x[i*10+7]=random.randrange(10,15)
    x[i*10+8]=random.randrange(5,10)
    x[i*10+9]=random.randrange(0,5)
x1 = np.arange(0,T-10)
plt.plot(x1,smooth(x,10))
plt.show()