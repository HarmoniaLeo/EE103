import numpy as np
from lagb import *
import random
import math
import matplotlib.pyplot as plt

colors=["b","r","y"]
m=5
h=0.1
eta=0.05
T=100
omegas=[[0,0],[-10,0],[-10,-10]]
pts=[[100,0],[200,0],[300,0]]
p0=np.array([0,0])
g=-9.8
count=1
plt.figure()
for omega in omegas:
    A=np.array([[1,0,h,0],[0,1,0,h],[0,0,1-h*eta/m,0],[0,0,0,1-h*eta/m]])
    b=np.array([0,0,h*eta*omega[0]/m,h*g+h*eta*omega[1]/m])
    F=A
    for i in range(1,T):
        F=dot(F,A)
    F11=F[:2,:2]
    F12=F[:2,2:]
    j=np.zeros(4)
    for i in range(0,T):
        jtmp=b
        for k in range(0,i):
            jtmp=dot(A,jtmp)
        j+=jtmp
    j1=j[:2]
    C=F12
    d=dot(F11,p0)+j1
    count2=0
    for pt in pts:
        v0=dot(rev(C),pt-d)
        x=np.array([0,0,v0[0],v0[1]])
        xs=[x[0]]
        ys=[x[1]]
        for i in range(0,T):
            x=dot(A,x)+b
            xs.append(x[0])
            ys.append(x[1])
        plt.subplot(3, 1, count)
        plt.plot(xs,ys,color=colors[count2],label="pt={0} v0={1}".format(pt,v0))
        count2+=1
    plt.subplot(3, 1, count).set_title("ω={0}".format(omega))
    plt.subplot(3, 1, count).legend()
    count+=1
plt.show()