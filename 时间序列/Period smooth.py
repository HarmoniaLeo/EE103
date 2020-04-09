import numpy as np
from scipy import linalg
import random
import matplotlib.pyplot as plt

l=1
T=200
P=10
n=int(T/P)
#上面这些是参数

x=np.empty((T,))
for i in range(0,n):
    x[i*10]=random.randrange(-10,10)
    x[i*10+1]=random.randrange(-15,15)
    x[i*10+2]=random.randrange(-20,20)
    x[i*10+3]=random.randrange(-25,25)
    x[i*10+4]=random.randrange(-30,30)
    x[i*10+5]=random.randrange(-25,25)
    x[i*10+6]=random.randrange(-20,20)
    x[i*10+7]=random.randrange(-15,15)
    x[i*10+8]=random.randrange(-10,10)
    x[i*10+9]=random.randrange(-5,5)
print("生成完毕")
A=np.zeros((P,P))
b=np.zeros((P,))
for i in range(0,P):
    if i==0:
        A[P-1][0]+=-l
    else:
        A[i-1][i]+=-l
    if i==P-1:
        A[0][P-1]+=-l
    else:
        A[i+1][i]+=-l
    A[i][i]+=n+2*l
    for j in range(0,n):
        b[i]+=x[j*P+i]
z=linalg.solve(A,b)
res=z
for i in range(0,n-1):
    res=np.concatenate((res,z),axis=0)
x1 = np.arange(0,T)
plt.plot(x1,x,label='original')	 
plt.plot(x1,res,label='smoothed',color='y')
plt.legend()
plt.show()