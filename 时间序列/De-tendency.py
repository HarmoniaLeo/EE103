import numpy as np
import random
import matplotlib.pyplot as plt
from lagb import *

T=200	#总时长
P=10	#周期长度
n=int(T/P)	#周期个数

def detendency(x,k):    #每k时间点计算一次趋势
    for i in range(0,int(x.shape[0]/k)):
        z1 = np.polyfit(np.linspace(0,k,k), x[i*k:(i+1)*k], 1)
        z = np.polyval(z1, np.linspace(0,k,k))
        x[i*k:(i+1)*k]=x[i*k:(i+1)*k]-z
    return x

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
x1 = np.arange(0,T)
plt.plot(x1,detendency(x,5))
plt.show()